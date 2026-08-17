export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

#setting
DATASET="${DATASET:-Instruments}"
TAIL_RECURRENCE_GAMMA="${TAIL_RECURRENCE_GAMMA:-0.4}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
EPOCHS="${EPOCHS:-200}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-256}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
# Directory containing the pretrained T5 model files.
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/models/t5-small}"
DATA_PATH="${DATA_PATH:-./data}"

INDEX_FILE_DEFAULT="$DATA_PATH/$DATASET/${DATASET}.index.json"

INDEX_FILE="${INDEX_FILE:-$INDEX_FILE_DEFAULT}"
if [[ ! -f "$INDEX_FILE" ]]; then
    echo "No index file found for dataset $DATASET at $INDEX_FILE" >&2
    exit 1
fi


GAMMA_FMT=$(printf "%.2f" "$TAIL_RECURRENCE_GAMMA")
GAMMA_FMT_SAFE="${GAMMA_FMT//./_}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/finetuned_ckpt/${DATASET}}"
mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/stage1_base/finetune.py" \
    --seed 42 \
    --finetuned_output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --index_file "$INDEX_FILE" \
    --temperature 1.0 \
    --base_model "$BASE_MODEL" \
    --c1_max_k 5 \
    --c1_enable_refine \
    --c1_tail_recurrence_gamma "$TAIL_RECURRENCE_GAMMA"
