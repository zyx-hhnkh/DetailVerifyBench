cd "$(dirname "$0")/../.."

JSONL_DIR=PATH_TO_HALLOC_INJECT_JSONL  # path to halloc inject jsonl
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select gpt-5.2 \
    --exp_name halloc_inject_gpt52_think \
    --use_think \
    --work_dir ./
