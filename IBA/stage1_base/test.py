import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
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
from iba_stage1 import IBAStage1


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

    tokenizer = T5Tokenizer.from_pretrained(args.finetuned_ckpt_path)
    _ = T5Config.from_pretrained(args.finetuned_ckpt_path)

    model = IBAStage1.from_pretrained(args.finetuned_ckpt_path, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
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
        "test_prompt_ids": args.test_prompt_ids,
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
