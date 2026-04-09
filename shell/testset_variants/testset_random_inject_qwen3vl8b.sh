cd "$(dirname "$0")/../.."

JSONL_DIR=PATH_TO_RANDOM_INJECT_JSONL  # path to random inject jsonl
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory
MODEL_DIR=PATH_TO_MODEL          # path to local model directory

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name random_inject_qwen3vl8b_think \
    --use_think \
    --work_dir ./
