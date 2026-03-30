cd "$(dirname "$0")/../.."

JSONL_DIR=/mnt/sdc/zhangyuxuan/halloc_inject/halloc_inject_results.jsonl
IMAGE_DIR=./ms_data/test
MODEL_DIR=/mnt/sdc/model_zoo/Qwen3-VL-8B-Thinking

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name halloc_inject_qwen3vl8b_think \
    --use_think \
    --work_dir ./
