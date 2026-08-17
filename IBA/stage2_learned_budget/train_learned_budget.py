import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os

os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import transformers
from transformers import EarlyStoppingCallback, TrainerCallback, T5Config, T5Tokenizer

from args import *
from iba_stage2 import IBAStage2
from IBA.common.collator import Collator


class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.loss_log = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        self.loss_log.append({
            "step": state.global_step,
            "train_loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
        })

    def on_train_end(self, args, state, control, **kwargs):
        import csv

        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "eval_loss"])
            writer.writeheader()
            writer.writerows(self.loss_log)
        print(f"loss log saved: {self.log_path}")


def _parse_int_list(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _has_model_weights(path: str) -> bool:
    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any(os.path.exists(os.path.join(path, name)) for name in weight_files)


def _resolve_pretrained_ckpt(path: str) -> str:
    if _has_model_weights(path):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = []
    for name in os.listdir(path):
        if not name.startswith("checkpoint-"):
            continue
        ckpt_dir = os.path.join(path, name)
        if not os.path.isdir(ckpt_dir) or not _has_model_weights(ckpt_dir):
            continue
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            step = -1
        candidates.append((step, ckpt_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No loadable weights found under {path}. "
            f"Expected model weights in the directory itself or in checkpoint-* subdirectories."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _count_trainable(model: IBAStage2):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_learned_budget(args):
    set_seed(args.seed)
    ensure_dir(args.finetuned_output_dir)

    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(vars(args))

    baseline_ckpt_path = _resolve_pretrained_ckpt(args.baseline_ckpt_path)
    if local_rank == 0 and baseline_ckpt_path != args.baseline_ckpt_path:
        print(f"resolved baseline checkpoint: {baseline_ckpt_path}")

    tokenizer = T5Tokenizer.from_pretrained(baseline_ckpt_path, model_max_length=512)
    config = T5Config.from_pretrained(baseline_ckpt_path)

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    if add_num != 0:
        raise ValueError(
            f"Tokenizer from baseline ckpt is not aligned with dataset tokens. "
            f"Unexpected newly added tokens: {add_num}"
        )

    budget_prior_schedule = _parse_int_list(getattr(args, "c1_budget_prior_schedule", None))
    if not budget_prior_schedule:
        raise ValueError("train_learned_budget requires --c1_budget_prior_schedule to define semantic ID positions, e.g. 3,2,1,0")
    if min(budget_prior_schedule) < 0:
        raise ValueError(f"c1_budget_prior_schedule must be >= 0. Got {budget_prior_schedule}")
    semantic_positions = len(budget_prior_schedule)

    required_max_k = int(getattr(args, "c1_learned_budget_kmax", 0))
    requested_max_k = int(getattr(args, "c1_max_k", 0))
    config.c1_max_k = max(int(getattr(config, "c1_max_k", 0)), requested_max_k, required_max_k)
    config.c1_enable_learned_budget = True
    config.c1_learned_budget_total = int(getattr(args, "c1_learned_budget_total", 6))
    config.c1_learned_budget_kmax = int(getattr(args, "c1_learned_budget_kmax", required_max_k))
    config.c1_budget_loss_weight = float(getattr(args, "c1_budget_loss_weight", 1.0))
    config.c1_budget_hidden_dim = int(getattr(args, "c1_budget_hidden_dim", 256))
    config.c1_budget_prior_weight = float(getattr(args, "c1_budget_prior_weight", 0.1))
    config.c1_budget_prior_schedule = budget_prior_schedule
    config.c1_budget_monotonic = bool(getattr(args, "c1_budget_monotonic", False))
    config.c1_semantic_positions = semantic_positions
    config.c1_commit_lookahead_weight = float(getattr(args, "c1_commit_lookahead_weight", 0.0))
    config.c1_commit_lookahead_max_offset = int(
        getattr(args, "c1_commit_lookahead_max_offset", 2)
    )
    config.c1_tail_recurrence_gamma = float(getattr(args, "c1_tail_recurrence_gamma", 0.5))
    config.use_cache = False

    if local_rank == 0:
        tokenizer.save_pretrained(args.finetuned_output_dir)
        config.save_pretrained(args.finetuned_output_dir)

    collator = Collator(args, tokenizer)

    model = IBAStage2.from_pretrained(
        baseline_ckpt_path,
        config=config,
    )
    model.reset_refinement_parameters()
    model._reset_budget_parameters()
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.set_learned_budget_enabled(True)
    model.config.use_cache = False
    model.to(device)

    trainable_params, total_params = _count_trainable(model)
    if local_rank == 0:
        print(f"learned-budget learned-budget trainable params: {trainable_params} / {total_params}")

    loss_callback = LossLoggingCallback(os.path.join(args.finetuned_output_dir, "loss_log.csv"))
    max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))

    training_args = transformers.TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_step,
        optim=args.optim,
        eval_strategy=args.save_and_eval_strategy,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.finetuned_output_dir,
        save_total_limit=2,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="none",
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        max_grad_norm=max_grad_norm,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20), loss_callback],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.finetuned_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBA_learned_budget_train")
    parser.add_argument("--baseline_ckpt_path", type=str, required=True, help="trained baseline checkpoint path")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    train_learned_budget(args)
