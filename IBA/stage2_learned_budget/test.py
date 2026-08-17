import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import datetime
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config

from args import *
from IBA.common.collator import TestCollator
from IBA.common.evaluate import get_topk_results, get_metrics_results
from IBA.common.generation_trie import Trie
from iba_stage2 import IBAStage2


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


def prefix_allowed_tokens_fn(trie: Trie):
    def _prefix_allowed_tokens(batch_id, input_ids):
        prefix = input_ids.tolist()
        return trie.get(prefix)
    return _prefix_allowed_tokens


def test(args):
    set_seed(args.seed)
    print(vars(args))

    gpu_id = getattr(args, "gpu_id", 0)
    device = torch.device("cuda", gpu_id)
    ckpt_path = _resolve_pretrained_ckpt(args.finetuned_ckpt_path)
    if ckpt_path != args.finetuned_ckpt_path:
        print(f"resolved checkpoint: {ckpt_path}")

    # load tokenizer/config from finetuned ckpt (vocab alignment)
    tokenizer = T5Tokenizer.from_pretrained(ckpt_path)
    ckpt_config = T5Config.from_pretrained(ckpt_path)

    model = IBAStage2.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()

    # ========================= 修改点开始 =========================
    # 删除“从 args 覆盖 k / alpha / enable_refine / use_cache”等逻辑
    # 说明：LETTER.from_pretrained 会自动从 ckpt/config.json 读取并初始化这些超参，
    #      因此 test 阶段直接使用 ckpt 里保存的配置即可，避免人为覆盖。
    # ========================= 修改点结束 =========================

    prompt_ids = [0]

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    metrics = args.metrics.split(",")
    all_prompt_results = []
    max_new_tokens = getattr(args, "max_new_tokens", 10)

    with torch.no_grad():
        for prompt_id in prompt_ids:
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for _, batch in enumerate(tqdm(test_loader)):
                inputs, targets = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                total += len(targets)

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )

                output_ids = output.sequences
                scores = output.sequences_scores
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                topk_res = get_topk_results(
                    decoded,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if args.filter_items else None,
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    metrics_results[m] = metrics_results.get(m, 0) + res

            for m in metrics_results:
                metrics_results[m] /= total

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print(f"Prompt {prompt_id} results:", metrics_results)
            print("======================================================\n")

    mean_results = {}
    min_results = {}
    max_results = {}
    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results:", mean_results)
    print("Min results:", min_results)
    print("Max results:", max_results)
    print("======================================================")

    save_data = {
        "saved_at_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolved_ckpt_path": ckpt_path,
        "results_file": args.results_file,
        "test_config": vars(args),
        "model_config": ckpt_config.to_dict(),
        "test_prompt_ids": args.test_prompt_ids,
        "metrics": metrics,
        "mean_results": mean_results,
        "min_results": min_results,
        "max_results": max_results,
        "all_prompt_results": all_prompt_results,
    }
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()
    test(args)
