#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${DATASET:-Microlens}"
# Dataset directory containing <DATASET>.item.json, <DATASET>.index.json, and <DATASET>.inter.json.
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data}"
# Fine-tuned model checkpoint directory to evaluate.
CKPT_PATH="$ROOT_DIR/finetuned_ckpt/Microlens"
INDEX_FILE_DEFAULT="$ROOT_DIR/data/${DATASET}/${DATASET}.index.json"
INDEX_FILE="${INDEX_FILE:-$INDEX_FILE_DEFAULT}"

if [[ ! -f "$INDEX_FILE" ]]; then
    echo "No index file found for dataset $DATASET at $INDEX_FILE" >&2
    exit 1
fi

ckpt_name="$(basename "$CKPT_PATH")"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_FILE:-$ROOT_DIR/results/${DATASET}/${ckpt_name}_${timestamp}.json}"
mkdir -p "$(dirname "$RESULTS_FILE")"

python3 "$SCRIPT_DIR/stage2_learned_budget/test.py" \
    --gpu_id "${GPU_ID:-1}" \
    --seed "${SEED:-42}" \
    --finetuned_ckpt_path "$CKPT_PATH" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --results_file "$RESULTS_FILE" \
    --test_batch_size "${TEST_BATCH_SIZE:-32}" \
    --num_beams "${NUM_BEAMS:-20}" \
    --test_prompt_ids "${TEST_PROMPT_IDS:-0}" \
    --index_file "$INDEX_FILE"
