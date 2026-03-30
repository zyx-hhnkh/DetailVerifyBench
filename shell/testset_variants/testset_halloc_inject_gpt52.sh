cd "$(dirname "$0")/../.."

JSONL_DIR=/mnt/sdc/zhangyuxuan/halloc_inject/halloc_inject_results.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select gpt-5.2 \
    --exp_name halloc_inject_gpt52_think \
    --use_think \
    --work_dir ./
