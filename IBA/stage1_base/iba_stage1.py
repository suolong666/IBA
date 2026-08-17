from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

import copy

logger = logging.get_logger(__name__)

DEFAULT_ENABLE_REFINE = True
DEFAULT_MAX_K = 2
DEFAULT_K = 2

DEFAULT_ENABLE_VERTICAL = True
DEFAULT_VERT_LAYERS = 5

DEFAULT_ENABLE_TAIL_IN_RECURRENCE = True
DEFAULT_TAIL_RECURRENCE_GAMMA = 0.5

def _parse_alphas_str(s: str):
    if s is None: return None
    s = str(s).strip()
    if s == "": return None
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def _default_alphas(k: int):
    if k <= 0: return []
    if k == 1: return [0.2]
    if k == 2: return [0.3, 0.6]
    return [min(0.2 * (i + 1), 0.8) for i in range(k)]

class IBAStage1(T5ForConditionalGeneration):
    """
    IBAStage1: Horizontal Hidden Refinement + Vertical Decoder Deepening.
    Integrated with Scheme C: Stage-Specific FiLM Modulation.

    This scheme modulates the vertical tail's output using learned affine
    parameters (gamma, beta) specific to each refinement stage[cite: 1, 41, 54].
    """

    def __init__(self, config: T5Config):
        super().__init__(config)

        self.temperature = 1.0

        self.enable_c1_refine = bool(getattr(config, "c1_enable_refine", DEFAULT_ENABLE_REFINE))
        self.c1_max_k = int(getattr(config, "c1_max_k", DEFAULT_MAX_K))
        self.c1_k = int(getattr(config, "c1_k", DEFAULT_K))
        if self.c1_k > self.c1_max_k: self.c1_k = self.c1_max_k
        cfg_alphas = getattr(config, "c1_alphas", None)
        if isinstance(cfg_alphas, (list, tuple)) and len(cfg_alphas) > 0:
            self.c1_alphas = [float(x) for x in cfg_alphas]
        else:
            parsed = _parse_alphas_str(cfg_alphas) if isinstance(cfg_alphas, str) else None
            self.c1_alphas = parsed if parsed is not None else _default_alphas(self.c1_max_k)

        self.c1_step_embeds = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.d_model)) for _ in range(self.c1_max_k)]
        )
        for p in self.c1_step_embeds: nn.init.normal_(p, mean=0.0, std=0.02)

        self.enable_c1_vertical = bool(getattr(config, "c1_enable_vertical", DEFAULT_ENABLE_VERTICAL))
        self.c1_vert_layers = int(getattr(config, "c1_vert_layers", DEFAULT_VERT_LAYERS))

        self.enable_c1_tail_in_recurrence = bool(
            getattr(config, "c1_enable_tail_in_recurrence", DEFAULT_ENABLE_TAIL_IN_RECURRENCE)
        )
        self.c1_tail_recurrence_gamma = float(
            getattr(config, "c1_tail_recurrence_gamma", DEFAULT_TAIL_RECURRENCE_GAMMA)
        )

        self.c1_vertical_decoder = None
        if self.enable_c1_vertical and self.c1_vert_layers > 0:
            vert_cfg = copy.deepcopy(config)
            vert_cfg.is_decoder = True
            vert_cfg.is_encoder_decoder = True
            vert_cfg.num_layers = self.c1_vert_layers
            self.c1_vertical_decoder = T5Stack(vert_cfg, embed_tokens=self.shared)

            if hasattr(self.decoder, "block") and len(self.decoder.block) > 0:
                proto = self.decoder.block[-1]
                self.c1_vertical_decoder.block = nn.ModuleList(
                    [copy.deepcopy(proto) for _ in range(self.c1_vert_layers)]
                )
            if hasattr(self.decoder, "final_layer_norm") and hasattr(self.c1_vertical_decoder, "final_layer_norm"):
                self.c1_vertical_decoder.final_layer_norm.load_state_dict(self.decoder.final_layer_norm.state_dict())

            self.film_generator = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_model // 4),
                    nn.ReLU(),
                    nn.Linear(config.d_model // 4, 2 * config.d_model)
                ) for _ in range(self.c1_max_k + 1)
            ])

            for gen in self.film_generator:
                nn.init.zeros_(gen[-1].weight)
                nn.init.zeros_(gen[-1].bias)

            logger.info(f"[IBAStage1] Scheme C Initialized: Stage-Specific FiLM. Layers={self.c1_vert_layers}")

    def set_hyper(self, temperature): self.temperature = temperature
    def set_refine_enabled(self, enabled: bool): self.enable_c1_refine = bool(enabled)
    def set_refine_k(self, k: int):
        self.c1_k = min(int(k), self.c1_max_k)
    def set_refine_alphas(self, alphas): self.c1_alphas = [float(x) for x in alphas]

    def ranking_loss(self, lm_logits, labels):
        if labels is None: return None
        t_logits = lm_logits / self.temperature
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        return loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.to(lm_logits.device).view(-1))

    def total_loss(self, lm_logits, labels, decoder_input_ids):
        return self.ranking_loss(lm_logits, labels)

    def _tail_depth_for_stage(self, stage_idx: int) -> int:
        return stage_idx

    def _apply_vertical_tail(
        self,
        hidden_states: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_head_mask: torch.Tensor,
        cross_attn_head_mask: torch.Tensor,
        return_dict: bool,
        depth: int, # stage_idx
    ) -> torch.Tensor:
        if (self.c1_vertical_decoder is None) or (not self.enable_c1_vertical):
            return hidden_states

        stage_idx = depth

        out = self.c1_vertical_decoder(
            input_ids=None, inputs_embeds=hidden_states, attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=False, return_dict=True
        )
        H_tail = out.last_hidden_state

        modulation = self.film_generator[stage_idx](hidden_states)
        gamma, beta = torch.chunk(modulation, 2, dim=-1)

        return (1.0 + gamma) * H_tail + beta

    def _logits_from_tail_hidden(self, H_tail: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            H_tail = H_tail * (self.model_dim ** -0.5)
        return self.lm_head(H_tail)

    def forward(
        self,
        input_ids=None, attention_mask=None, encoder_outputs=None,
        decoder_input_ids=None, decoder_attention_mask=None,
        past_key_values=None, use_cache=None, labels=None,
        inputs_embeds=None, decoder_inputs_embeds=None,
        return_dict=None, **kwargs,
    ):
        if use_cache is None: use_cache = self.config.use_cache
        if return_dict is None: return_dict = self.config.use_return_dict

        if self.enable_c1_refine and self.c1_k > 0 and labels is None:
            use_cache = False

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        dec_out = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask,
            use_cache=use_cache, return_dict=return_dict,
        )
        H0_raw = dec_out.last_hidden_state
        H0_tail = self._apply_vertical_tail(H0_raw, decoder_attention_mask, encoder_hidden_states, attention_mask, None, None, return_dict, depth=0)
        logits0 = self._logits_from_tail_hidden(H0_tail)

        logits_list = [logits0]; loss_list = []
        if labels is not None: loss_list.append(self.ranking_loss(logits0, labels))

        H_prev_raw = H0_raw
        if self.enable_c1_refine and self.c1_k > 0:
            for j in range(1, self.c1_k + 1):
                if self.enable_c1_tail_in_recurrence:
                    H_prev_tail = self._apply_vertical_tail(H_prev_raw, decoder_attention_mask, encoder_hidden_states, attention_mask, None, None, return_dict, depth=j-1)
                    H_prev_in = (1.0 - self.c1_tail_recurrence_gamma) * H_prev_raw + self.c1_tail_recurrence_gamma * H_prev_tail
                else:
                    H_prev_in = H_prev_raw

                H_in = H_prev_in + self.c1_step_embeds[j-1].view(1, 1, -1).to(H_prev_in.device)
                decj = self.decoder(inputs_embeds=H_in, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, use_cache=False)
                Hj_raw = decj.last_hidden_state
                H_prev_raw = Hj_raw

                Hj_tail = self._apply_vertical_tail(Hj_raw, decoder_attention_mask, encoder_hidden_states, attention_mask, None, None, return_dict, depth=j)
                logitsj = self._logits_from_tail_hidden(Hj_tail)
                logits_list.append(logitsj)
                if labels is not None: loss_list.append(self.ranking_loss(logitsj, labels))

        lm_logits = logits_list[-1]
        loss = None
        if labels is not None:
            total = loss_list[-1]
            for i in range(self.c1_k): total = total + float(self.c1_alphas[i]) * loss_list[i]
            loss = total

        return Seq2SeqLMOutput(loss=loss, logits=lm_logits, encoder_last_hidden_state=encoder_outputs.last_hidden_state)