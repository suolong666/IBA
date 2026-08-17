export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Dataset directory containing <DATASET>.item.json, <DATASET>.index.json, and <DATASET>.inter.json.
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data}"

# setting
DATASET="${DATASET:-Microlens}"
# Stage-1 fine-tuned checkpoint directory used to initialize stage 2.
BASELINE_CKPT="${BASELINE_CKPT:-$ROOT_DIR/finetuned_ckpt/Microlens}"
BASE_MODEL="${BASE_MODEL:-$BASELINE_CKPT}"
MAX_PRIVATE_K="${MAX_PRIVATE_K:-3}"
LEARNED_BUDGET_TOTAL="${LEARNED_BUDGET_TOTAL:-6}"
BUDGET_LOSS_WEIGHT="${BUDGET_LOSS_WEIGHT:-1.0}"
BUDGET_HIDDEN_DIM="${BUDGET_HIDDEN_DIM:-256}"
BUDGET_PRIOR_WEIGHT="${BUDGET_PRIOR_WEIGHT:-0.1}"
BUDGET_PRIOR_SCHEDULE="${BUDGET_PRIOR_SCHEDULE:-3,2,1,0}"
# Set to 1/true to restrict feasible budgets to non-increasing schedules.
BUDGET_MONOTONIC="${BUDGET_MONOTONIC:-1}"
LOOKAHEAD_WEIGHT="${LOOKAHEAD_WEIGHT:-0.15}"
LOOKAHEAD_MAX_OFFSET="${LOOKAHEAD_MAX_OFFSET:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
EPOCHS="${EPOCHS:-50}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-128}"
TAIL_RECURRENCE_GAMMA="${TAIL_RECURRENCE_GAMMA:-0.5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
gradient_accumulation_steps=4

# Dataset index JSON file for the selected dataset.
INDEX_FILE_DEFAULT="$ROOT_DIR/data/Microlens/Microlens.index.json"
INDEX_FILE="${INDEX_FILE:-$INDEX_FILE_DEFAULT}"
if [[ ! -f "$INDEX_FILE" ]]; then
    echo "No index file found for dataset $DATASET at $INDEX_FILE" >&2
    exit 1
fi


prior_suffix="${BUDGET_PRIOR_SCHEDULE//,/}"
case "${BUDGET_MONOTONIC,,}" in
    1|true|yes|on) BUDGET_MONOTONIC_FLAG=1 ;;
    *) BUDGET_MONOTONIC_FLAG=0 ;;
esac
lookahead_tag=$(printf '%03d' "$(LOOKAHEAD_WEIGHT="$LOOKAHEAD_WEIGHT" python3 - <<'PY'
import os
value = float(os.environ["LOOKAHEAD_WEIGHT"])
print(int(round(value * 100)))
PY
)")
gamma_tag=$(printf '%.2f' "$TAIL_RECURRENCE_GAMMA")
gamma_tag_safe="${gamma_tag//./_}"
suffix="learnedB${LEARNED_BUDGET_TOTAL}_K${MAX_PRIVATE_K}_prior${prior_suffix}_mono${BUDGET_MONOTONIC_FLAG}_la${lookahead_tag}_g${gamma_tag_safe}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/finetuned_ckpt/${DATASET}_${suffix}}"
mkdir -p "$OUTPUT_DIR"

args=(
    --seed 42
    --baseline_ckpt_path "$BASELINE_CKPT"
    --finetuned_output_dir "$OUTPUT_DIR"
    --dataset "$DATASET"
    --data_path "$DATA_PATH"
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"

    --learning_rate "$LEARNING_RATE"
    --warmup_ratio "$WARMUP_RATIO"
    --max_grad_norm "$MAX_GRAD_NORM"
    --epochs "$EPOCHS"
    --index_file "$INDEX_FILE"
    --temperature 1.0
    --base_model "$BASE_MODEL"
    --c1_max_k "$MAX_PRIVATE_K"
    --c1_enable_learned_budget
    --c1_learned_budget_total "$LEARNED_BUDGET_TOTAL"
    --c1_learned_budget_kmax "$MAX_PRIVATE_K"
    --c1_budget_loss_weight "$BUDGET_LOSS_WEIGHT"
    --c1_budget_hidden_dim "$BUDGET_HIDDEN_DIM"
    --c1_budget_prior_weight "$BUDGET_PRIOR_WEIGHT"
    --c1_budget_prior_schedule "$BUDGET_PRIOR_SCHEDULE"
    --c1_commit_lookahead_weight "$LOOKAHEAD_WEIGHT"
    --c1_commit_lookahead_max_offset "$LOOKAHEAD_MAX_OFFSET"
    --c1_tail_recurrence_gamma "$TAIL_RECURRENCE_GAMMA"
    --gradient_accumulation_steps 4
)

if [[ "$BUDGET_MONOTONIC_FLAG" == "1" ]]; then
    args+=(--c1_budget_monotonic)
fi

python3 "$SCRIPT_DIR/stage2_learned_budget/train_learned_budget.py" "${args[@]}"
