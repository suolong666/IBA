from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

import copy

logger = logging.get_logger(__name__)

DEFAULT_MAX_K = 3
DEFAULT_COMMIT_LOOKAHEAD_WEIGHT = 0.0
DEFAULT_COMMIT_LOOKAHEAD_MAX_OFFSET = 2
DEFAULT_DETACH_RECURRENCE = False
DEFAULT_AUX_DETACH_BACKBONE = False
DEFAULT_ENABLE_LEARNED_BUDGET = False
DEFAULT_LEARNED_BUDGET_TOTAL = 6
DEFAULT_LEARNED_BUDGET_KMAX = 3
DEFAULT_BUDGET_LOSS_WEIGHT = 1.0
DEFAULT_BUDGET_HIDDEN_DIM = 256
DEFAULT_BUDGET_PRIOR_WEIGHT = 0.1
DEFAULT_BUDGET_PRIOR_SCHEDULE = [3, 2, 1, 0]

DEFAULT_ENABLE_VERTICAL = True
DEFAULT_VERT_LAYERS = 5
DEFAULT_ENABLE_TAIL_IN_RECURRENCE = True
DEFAULT_TAIL_RECURRENCE_GAMMA = 0.5

def _parse_ints_str(s: str):
    if s is None: return None
    s = str(s).strip()
    if s == "": return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


class IBAStage2(T5ForConditionalGeneration):
    """
    IBAStage2: Horizontal Hidden Refinement + Vertical Decoder Deepening.
    Integrated with Scheme C: Stage-Specific FiLM Modulation.

    This scheme modulates the vertical tail's output using learned affine
    parameters (gamma, beta) specific to each refinement stage[cite: 1, 41, 54].
    """

    def __init__(self, config: T5Config):
        super().__init__(config)

        self.temperature = 1.0

        self.c1_max_k = int(getattr(config, "c1_max_k", DEFAULT_MAX_K))
        self.c1_commit_lookahead_weight = float(
            getattr(config, "c1_commit_lookahead_weight", DEFAULT_COMMIT_LOOKAHEAD_WEIGHT)
        )
        self.c1_commit_lookahead_max_offset = int(
            getattr(config, "c1_commit_lookahead_max_offset", DEFAULT_COMMIT_LOOKAHEAD_MAX_OFFSET)
        )
        self.enable_c1_learned_budget = bool(
            getattr(config, "c1_enable_learned_budget", DEFAULT_ENABLE_LEARNED_BUDGET)
        )
        self.c1_learned_budget_total = int(
            getattr(config, "c1_learned_budget_total", DEFAULT_LEARNED_BUDGET_TOTAL)
        )
        self.c1_learned_budget_kmax = int(
            getattr(config, "c1_learned_budget_kmax", min(DEFAULT_LEARNED_BUDGET_KMAX, self.c1_max_k))
        )
        self.c1_learned_budget_kmax = max(0, min(self.c1_learned_budget_kmax, self.c1_max_k))
        self.c1_budget_loss_weight = float(
            getattr(config, "c1_budget_loss_weight", DEFAULT_BUDGET_LOSS_WEIGHT)
        )
        self.c1_budget_hidden_dim = int(
            getattr(config, "c1_budget_hidden_dim", DEFAULT_BUDGET_HIDDEN_DIM)
        )
        self.c1_budget_prior_weight = float(
            getattr(config, "c1_budget_prior_weight", DEFAULT_BUDGET_PRIOR_WEIGHT)
        )
        self.c1_budget_monotonic = bool(getattr(config, "c1_budget_monotonic", False))

        cfg_prior_schedule = getattr(config, "c1_budget_prior_schedule", None)
        if isinstance(cfg_prior_schedule, (list, tuple)) and len(cfg_prior_schedule) > 0:
            self.c1_budget_prior_schedule = [int(x) for x in cfg_prior_schedule]
        else:
            parsed_prior = _parse_ints_str(cfg_prior_schedule) if isinstance(cfg_prior_schedule, str) else None
            self.c1_budget_prior_schedule = parsed_prior if parsed_prior is not None else list(DEFAULT_BUDGET_PRIOR_SCHEDULE)

        cfg_semantic_positions = getattr(config, "c1_semantic_positions", None)
        self.c1_semantic_positions = (
            max(int(cfg_semantic_positions), 1)
            if cfg_semantic_positions is not None
            else max(len(self.c1_budget_prior_schedule), 1)
        )

        self.c1_budget_position_embeds = nn.Embedding(self.c1_semantic_positions, config.d_model)
        self.c1_budget_step_embeds = nn.Embedding(max(self.c1_learned_budget_kmax, 1), config.d_model)
        self.c1_budget_predictor = nn.Sequential(
            nn.Linear(3 * config.d_model, self.c1_budget_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.c1_budget_hidden_dim, 1),
        )
        self._reset_budget_parameters()
        self._feasible_budget_cache = {}

        self.c1_step_embeds = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.d_model)) for _ in range(self.c1_max_k)]
        )
        for p in self.c1_step_embeds:
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.c1_commit_lookahead_offset_embeds = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.d_model)) for _ in range(self.c1_commit_lookahead_max_offset)]
        )
        for p in self.c1_commit_lookahead_offset_embeds:
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.enable_c1_vertical = bool(getattr(config, "c1_enable_vertical", DEFAULT_ENABLE_VERTICAL))
        self.c1_vert_layers = int(getattr(config, "c1_vert_layers", DEFAULT_VERT_LAYERS))

        self.enable_c1_tail_in_recurrence = bool(
            getattr(config, "c1_enable_tail_in_recurrence", DEFAULT_ENABLE_TAIL_IN_RECURRENCE)
        )
        self.c1_tail_recurrence_gamma = float(
            getattr(config, "c1_tail_recurrence_gamma", DEFAULT_TAIL_RECURRENCE_GAMMA)
        )
        self.c1_detach_recurrence = bool(
            getattr(config, "c1_detach_recurrence", DEFAULT_DETACH_RECURRENCE)
        )
        self.c1_aux_detach_backbone = bool(
            getattr(config, "c1_aux_detach_backbone", DEFAULT_AUX_DETACH_BACKBONE)
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

            logger.info(f"[IBAStage2] Scheme C Initialized: Stage-Specific FiLM. Layers={self.c1_vert_layers}")

    def set_hyper(self, temperature):
        self.temperature = temperature

    def set_learned_budget_enabled(self, enabled: bool):
        self.enable_c1_learned_budget = bool(enabled)

    def _reset_budget_parameters(self):
        for module in self.c1_budget_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.c1_budget_position_embeds.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c1_budget_step_embeds.weight, mean=0.0, std=0.02)

    def reset_refinement_parameters(self):
        for p in self.c1_step_embeds:
            nn.init.normal_(p, mean=0.0, std=0.02)
        for p in self.c1_commit_lookahead_offset_embeds:
            nn.init.normal_(p, mean=0.0, std=0.02)

    def _semantic_position_count(self) -> int:
        return int(self.c1_semantic_positions)

    def _logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            hidden_states = hidden_states * (self.model_dim ** -0.5)
        return self.lm_head(hidden_states) / self.temperature

    def _encoder_user_context(self, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        if encoder_attention_mask is None:
            return encoder_hidden_states.mean(dim=1)
        mask = encoder_attention_mask.to(encoder_hidden_states.device).to(encoder_hidden_states.dtype).unsqueeze(-1)
        safe_hidden = torch.nan_to_num(encoder_hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        safe_hidden = safe_hidden.masked_fill(mask.eq(0), 0.0)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (safe_hidden * mask).sum(dim=1) / denom

    def _predict_budget_utilities(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        semantic_positions: int,
    ) -> torch.Tensor:
        kmax = max(0, min(self.c1_learned_budget_kmax, self.c1_max_k))
        if kmax <= 0:
            return encoder_hidden_states.new_zeros(encoder_hidden_states.size(0), semantic_positions, 0)

        if semantic_positions > self.c1_budget_position_embeds.num_embeddings:
            raise ValueError(
                f"semantic_positions={semantic_positions} exceeds budget position embeddings "
                f"({self.c1_budget_position_embeds.num_embeddings})."
            )

        q = self._encoder_user_context(encoder_hidden_states, encoder_attention_mask)
        batch_size, d_model = q.size()
        pos_ids = torch.arange(semantic_positions, device=q.device)
        step_ids = torch.arange(kmax, device=q.device)
        pos_emb = self.c1_budget_position_embeds(pos_ids).view(1, semantic_positions, 1, d_model)
        step_emb = self.c1_budget_step_embeds(step_ids).view(1, 1, kmax, d_model)
        q_exp = q.view(batch_size, 1, 1, d_model).expand(batch_size, semantic_positions, kmax, d_model)
        pos_exp = pos_emb.expand(batch_size, semantic_positions, kmax, d_model)
        step_exp = step_emb.expand(batch_size, semantic_positions, kmax, d_model)
        features = torch.cat([q_exp, pos_exp, step_exp], dim=-1)
        utilities = self.c1_budget_predictor(features).squeeze(-1)
        return F.softplus(utilities)

    def _enumerate_feasible_budgets(self, semantic_positions: int, kmax: int, total_budget: int):
        monotonic = bool(getattr(self, "c1_budget_monotonic", False))
        key = (int(semantic_positions), int(kmax), int(total_budget), monotonic)
        if key in self._feasible_budget_cache:
            return self._feasible_budget_cache[key]

        schedules = []
        current = []

        def rec(pos: int, remaining: int):
            if pos == semantic_positions:
                if remaining == 0:
                    schedules.append(tuple(current))
                return

            slots_left = semantic_positions - pos - 1
            lower = max(0, remaining - slots_left * kmax)
            upper = min(kmax, remaining)
            for value in range(upper, lower - 1, -1):
                if monotonic and current and value > current[-1]:
                    continue
                current.append(value)
                rec(pos + 1, remaining - value)
                current.pop()

        rec(0, total_budget)
        if not schedules:
            raise ValueError(
                f"No feasible learned-budget schedule for L={semantic_positions}, "
                f"K_max={kmax}, B={total_budget}, monotonic={monotonic}."
            )
        self._feasible_budget_cache[key] = schedules
        return schedules

    def _feasible_budget_tensor(self, semantic_positions: int, device) -> torch.Tensor:
        kmax = max(0, min(self.c1_learned_budget_kmax, self.c1_max_k))
        schedules = self._enumerate_feasible_budgets(
            semantic_positions=semantic_positions,
            kmax=kmax,
            total_budget=self.c1_learned_budget_total,
        )
        return torch.tensor(schedules, device=device, dtype=torch.long)

    def _budget_prior_tensor(self, semantic_positions: int, device) -> torch.Tensor:
        prior = list(self.c1_budget_prior_schedule) if self.c1_budget_prior_schedule else [0] * semantic_positions
        if len(prior) < semantic_positions:
            prior = prior + [0] * (semantic_positions - len(prior))
        prior = prior[:semantic_positions]
        kmax = max(0, min(self.c1_learned_budget_kmax, self.c1_max_k))
        prior = [max(0, min(int(x), kmax)) for x in prior]
        return torch.tensor(prior, device=device, dtype=torch.long)

    def _select_learned_budgets(self, utilities: torch.Tensor) -> torch.Tensor:
        batch_size, semantic_positions, kmax = utilities.size()
        if kmax == 0:
            return utilities.new_zeros(batch_size, semantic_positions, dtype=torch.long)
        schedules = self._feasible_budget_tensor(semantic_positions, utilities.device)
        step_ids = torch.arange(kmax, device=utilities.device).view(1, 1, kmax)
        schedule_mask = (step_ids < schedules.unsqueeze(-1)).to(utilities.dtype)
        scores = (utilities.unsqueeze(1) * schedule_mask.unsqueeze(0)).sum(dim=(-1, -2))
        prior_weight = float(getattr(self, "c1_budget_prior_weight", 0.0))
        if prior_weight > 0.0:
            prior = self._budget_prior_tensor(semantic_positions, utilities.device)
            prior_penalty = (schedules - prior.view(1, -1)).abs().sum(dim=1).to(scores.dtype)
            scores = scores - prior_weight * prior_penalty.view(1, -1)
        best_idx = scores.argmax(dim=1)
        return schedules.index_select(0, best_idx)

    def _ce_loss_per_sample(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.device)
        valid = labels.ne(-100)
        safe_labels = labels.masked_fill(~valid, 0)
        losses = F.cross_entropy(logits / self.temperature, safe_labels, reduction="none")
        return losses.masked_fill(~valid, 0.0)

    def _run_budget_refinement_position_all_depths(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        public_decoder_input_ids: torch.Tensor,
        public_decoder_attention_mask: torch.Tensor,
        return_dict: bool,
        max_private_steps: int,
    ):
        base_out = self.decoder(
            input_ids=public_decoder_input_ids,
            attention_mask=public_decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=return_dict,
        )
        hidden_seq = base_out.last_hidden_state
        logits_by_depth = []
        hidden_by_depth = []

        tail_seq = self._apply_vertical_tail(
            hidden_seq,
            public_decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            None,
            return_dict,
            depth=0,
        )
        logits_by_depth.append(self._logits_from_hidden(tail_seq[:, -1, :]))
        hidden_by_depth.append(tail_seq[:, -1, :])

        max_private_steps = max(0, min(int(max_private_steps), self.c1_max_k))
        if max_private_steps > 0:
            token_embeds = self.shared(public_decoder_input_ids)
            for step_idx in range(max_private_steps):
                iter_inputs = token_embeds.clone()
                iter_inputs[:, -1, :] = (
                    hidden_seq[:, -1, :] + self.c1_step_embeds[step_idx].view(1, -1).to(hidden_seq.device)
                )
                step_out = self.decoder(
                    inputs_embeds=iter_inputs,
                    attention_mask=public_decoder_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=False,
                    return_dict=return_dict,
                )
                hidden_seq = step_out.last_hidden_state
                tail_seq = self._apply_vertical_tail(
                    hidden_seq,
                    public_decoder_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    None,
                    return_dict,
                    depth=step_idx + 1,
                )
                logits_by_depth.append(self._logits_from_hidden(tail_seq[:, -1, :]))
                hidden_by_depth.append(tail_seq[:, -1, :])

        return torch.stack(logits_by_depth, dim=1), torch.stack(hidden_by_depth, dim=1)

    def _budget_refinement_lookahead_loss(
        self,
        hidden_last: torch.Tensor,
        labels: torch.Tensor,
        position_idx: int,
    ) -> torch.Tensor:
        if self.c1_commit_lookahead_weight <= 0.0:
            return hidden_last.sum() * 0.0

        semantic_positions = min(self._semantic_position_count(), labels.size(1))
        losses = []
        max_offset = min(self.c1_commit_lookahead_max_offset, len(self.c1_commit_lookahead_offset_embeds))
        for offset in range(1, max_offset + 1):
            future_pos = position_idx + offset
            if future_pos >= semantic_positions:
                break
            lookahead_hidden = hidden_last + self.c1_commit_lookahead_offset_embeds[offset - 1].view(1, -1)
            lookahead_logits = self._logits_from_hidden(lookahead_hidden)
            losses.append(self.ranking_loss(lookahead_logits.unsqueeze(1), labels[:, future_pos].unsqueeze(1)))

        if not losses:
            return hidden_last.sum() * 0.0
        return torch.stack(losses).mean()

    def _forward_learned_budget_training(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        batch_size, seq_len = decoder_input_ids.size()
        semantic_positions = min(self._semantic_position_count(), labels.size(1), seq_len)
        kmax = max(0, min(self.c1_learned_budget_kmax, self.c1_max_k))
        utilities = self._predict_budget_utilities(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            semantic_positions=semantic_positions,
        )
        learned_budgets = self._select_learned_budgets(utilities)

        logits_steps = []
        lookahead_losses = []
        budget_losses = []
        batch_idx = torch.arange(batch_size, device=decoder_input_ids.device)

        for pos in range(seq_len):
            public_decoder_input_ids = decoder_input_ids[:, : pos + 1]
            public_decoder_attention_mask = (
                decoder_attention_mask[:, : pos + 1] if decoder_attention_mask is not None else None
            )

            if pos < semantic_positions:
                logits_by_depth, hidden_by_depth = self._run_budget_refinement_position_all_depths(
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    public_decoder_input_ids=public_decoder_input_ids,
                    public_decoder_attention_mask=public_decoder_attention_mask,
                    return_dict=True,
                    max_private_steps=kmax,
                )
                selected_depth = learned_budgets[:, pos].clamp(min=0, max=kmax)
                logits_last = logits_by_depth[batch_idx, selected_depth, :]
                hidden_last = hidden_by_depth[batch_idx, selected_depth, :]
                logits_steps.append(logits_last.unsqueeze(1))

                label_pos = labels[:, pos]
                per_depth_losses = [
                    self._ce_loss_per_sample(logits_by_depth[:, depth_idx, :], label_pos)
                    for depth_idx in range(kmax + 1)
                ]
                per_depth_losses = torch.stack(per_depth_losses, dim=1)
                target_delta = (per_depth_losses[:, :-1] - per_depth_losses[:, 1:]).clamp(0.0, 20.0).detach()
                valid = label_pos.to(logits_by_depth.device).ne(-100)
                pred_delta = utilities[:, pos, :]
                if valid.any():
                    budget_losses.append(F.smooth_l1_loss(pred_delta[valid], target_delta[valid], reduction="mean"))
                else:
                    budget_losses.append(pred_delta.sum() * 0.0)

                lookahead_losses.append(
                    self._budget_refinement_lookahead_loss(
                        hidden_last=hidden_last,
                        labels=labels,
                        position_idx=pos,
                    )
                )
            else:
                logits_by_depth, _ = self._run_budget_refinement_position_all_depths(
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    public_decoder_input_ids=public_decoder_input_ids,
                    public_decoder_attention_mask=public_decoder_attention_mask,
                    return_dict=True,
                    max_private_steps=0,
                )
                logits_steps.append(logits_by_depth[:, 0, :].unsqueeze(1))

        lm_logits = torch.cat(logits_steps, dim=1)
        loss = self.ranking_loss(lm_logits, labels)
        if lookahead_losses and self.c1_commit_lookahead_weight > 0.0:
            loss = loss + self.c1_commit_lookahead_weight * torch.stack(lookahead_losses).mean()
        if budget_losses and self.c1_budget_loss_weight > 0.0:
            loss = loss + self.c1_budget_loss_weight * torch.stack(budget_losses).mean()

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            encoder_last_hidden_state=encoder_hidden_states,
        )

    def _forward_learned_budget_inference(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        batch_size, seq_len = decoder_input_ids.size()
        semantic_positions = self._semantic_position_count()
        position_idx = seq_len - 1
        kmax = max(0, min(self.c1_learned_budget_kmax, self.c1_max_k))

        if position_idx < semantic_positions:
            utilities = self._predict_budget_utilities(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                semantic_positions=semantic_positions,
            )
            learned_budgets = self._select_learned_budgets(utilities)
            logits_by_depth, _ = self._run_budget_refinement_position_all_depths(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                public_decoder_input_ids=decoder_input_ids,
                public_decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
                max_private_steps=kmax,
            )
            selected_depth = learned_budgets[:, position_idx].clamp(min=0, max=kmax)
            batch_idx = torch.arange(batch_size, device=decoder_input_ids.device)
            logits_last = logits_by_depth[batch_idx, selected_depth, :]
        else:
            logits_by_depth, _ = self._run_budget_refinement_position_all_depths(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                public_decoder_input_ids=decoder_input_ids,
                public_decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
                max_private_steps=0,
            )
            logits_last = logits_by_depth[:, 0, :]

        vocab_size = logits_last.size(-1)
        lm_logits = logits_last.new_zeros(batch_size, seq_len, vocab_size)
        lm_logits[:, -1, :] = logits_last
        return Seq2SeqLMOutput(
            loss=None,
            logits=lm_logits,
            encoder_last_hidden_state=encoder_hidden_states,
        )

    def ranking_loss(self, lm_logits, labels):
        if labels is None: return None
        t_logits = lm_logits / self.temperature
        flat_labels = labels.to(lm_logits.device).view(-1)
        if not flat_labels.ne(-100).any():
            return t_logits.sum() * 0.0
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        return loss_fct(t_logits.view(-1, t_logits.size(-1)), flat_labels)

    def _build_decoder_attention_mask(self, decoder_input_ids, decoder_attention_mask, labels):
        if decoder_attention_mask is not None:
            return decoder_attention_mask
        if decoder_input_ids is not None:
            return decoder_input_ids.ne(self.config.pad_token_id).long()
        if labels is not None:
            return labels.ne(-100).long()
        return None

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

        # 1. Base vertical deepening
        out = self.c1_vertical_decoder(
            input_ids=None, inputs_embeds=hidden_states, attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            use_cache=False, return_dict=True
        )
        H_tail = out.last_hidden_state

        # 2. Stage-specific FiLM Parameter Generation
        # modulation: [B, T, 2*d]
        modulation = self.film_generator[stage_idx](hidden_states)
        gamma, beta = torch.chunk(modulation, 2, dim=-1)

        # 3. Apply Modulation: (1 + gamma) * H + beta
        return (1.0 + gamma) * H_tail + beta

    def _logits_from_tail_hidden(self, H_tail: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            H_tail = H_tail * (self.model_dim ** -0.5)
        return self.lm_head(H_tail)

    # ---------- Forward ----------
    def forward(
        self,
        input_ids=None, attention_mask=None, encoder_outputs=None,
        decoder_input_ids=None, decoder_attention_mask=None,
        past_key_values=None, use_cache=None, labels=None,
        inputs_embeds=None, decoder_inputs_embeds=None,
        return_dict=None, **kwargs,
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict
        use_cache = False

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_input_ids is None:
            raise ValueError("learned-budget refinement requires decoder_input_ids")

        decoder_attention_mask = self._build_decoder_attention_mask(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        if not self.enable_c1_learned_budget:
            raise ValueError("stage2_learned_budget IBAStage2 requires c1_enable_learned_budget=True")

        if labels is not None:
            return self._forward_learned_budget_training(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
        return self._forward_learned_budget_inference(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
