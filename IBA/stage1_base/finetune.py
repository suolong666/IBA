import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

os.environ.setdefault("WANDB_MODE", "disabled")
import torch
import transformers
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers import T5Tokenizer, T5Config

from iba_stage1 import IBAStage1
from args import *
from IBA.common.collator import Collator


class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.loss_log = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.loss_log.append({
                "step": state.global_step,
                "train_loss": logs.get("loss", None),
                "eval_loss": logs.get("eval_loss", None)
            })

    def on_train_end(self, args, state, control, **kwargs):
        df = pd.DataFrame(self.loss_log)
        df.to_csv(self.log_path, index=False)
        print(f"loss log saved: {self.log_path}")
        self.plot_loss(df, self.log_path.replace(".csv", ".png"))

    def plot_loss(self, df, save_path):
        plt.figure(figsize=(10, 6))
        train = df[df["train_loss"].notna()]
        ev = df[df["eval_loss"].notna()]
        plt.plot(train["step"], train["train_loss"], label="Train Loss")
        plt.plot(ev["step"], ev["eval_loss"], label="Eval Loss", marker="o")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"loss curve saved: {save_path}")


def _parse_alphas(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def train(args):
    set_seed(args.seed)
    ensure_dir(args.finetuned_output_dir)

    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(vars(args))

    # ===== load base config/tokenizer =====
    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)

    # ===== datasets & vocab =====
    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    # ===== read args for refinement =====
    c1_max_k = int(getattr(args, "c1_max_k", 2))
    c1_k = int(getattr(args, "c1_k", 2))
    if c1_k > c1_max_k:
        raise ValueError(f"c1_k={c1_k} must be <= c1_max_k={c1_max_k}")

    c1_enable_refine = bool(getattr(args, "c1_enable_refine", True))
    alphas = _parse_alphas(getattr(args, "c1_alphas", None))
    if alphas is not None and len(alphas) < c1_k:
        raise ValueError(f"Need len(c1_alphas) >= c1_k. Got {len(alphas)} < {c1_k}")

    # write into config so ckpt remembers structure & defaults
    config.c1_max_k = c1_max_k
    config.c1_k = c1_k
    config.c1_enable_refine = c1_enable_refine
    if alphas is not None:
        config.c1_alphas = alphas
    # tail recurrence gamma controls blending of raw and tail hidden states
    config.c1_tail_recurrence_gamma = float(getattr(args, "c1_tail_recurrence_gamma", 0.5))

    if local_rank == 0:
        print(f"add {add_num} new token.")
        tokenizer.save_pretrained(args.finetuned_output_dir)
        config.save_pretrained(args.finetuned_output_dir)

    collator = Collator(args, tokenizer)

    # ===== model =====
    model = IBAStage1(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # override runtime settings explicitly
    model.set_refine_enabled(c1_enable_refine)
    model.set_refine_k(c1_k if c1_enable_refine else 0)
    if alphas is not None:
        model.set_refine_alphas(alphas)

    # correctness-first refinement: cache off
    model.config.use_cache = False

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
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="none",
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,
        max_grad_norm=max_grad_norm,  # gradient clipping
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
    parser = argparse.ArgumentParser(description="LLMRec")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    train(args)
