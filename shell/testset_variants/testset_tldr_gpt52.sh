cd "$(dirname "$0")/../.."

JSONL_DIR=/mnt/sdc/Hallucination_DATA/Hallucination_Bench/TLDR_testset_inject.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode tldr \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select gpt-5.2 \
    --exp_name tldr_injected_gpt52_think \
    --use_think \
    --work_dir ./
