cd "$(dirname "$0")/../.."

MODEL_DIR=/mnt/sdc/model_zoo/Qwen3-VL-8B-Thinking
JSONL_DIR=./ms_data/test/test_gt_withtag.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --mode offline \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name testset_Qwen3-VL-8B_think_0317 \
    --use_think \
    --work_dir ./ 

