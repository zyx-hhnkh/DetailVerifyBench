cd "$(dirname "$0")/../.."

JSONL_DIR=/mnt/sdc/Hallucination_DATA/Hallucination_Bench/testset_geminiflash_vs_gpt52_n2.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select gpt-5.4 \
    --exp_name injected_testset_gpt54_think \
    --use_think \
    --work_dir ./
