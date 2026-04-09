cd "$(dirname "$0")/../.."

JSONL_DIR=PATH_TO_TLDR_JSONL  # path to TLDR inject jsonl
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode tldr \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select gpt-5.2 \
    --exp_name tldr_injected_gpt52_think \
    --use_think \
    --work_dir ./
