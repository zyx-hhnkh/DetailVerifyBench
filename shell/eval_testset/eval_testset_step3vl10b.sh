cd "$(dirname "$0")/../.."

MODEL_DIR=/mnt/sdc/model_zoo/Step3-VL-10B
JSONL_DIR=./ms_data/test/test_gt_withtag.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select Step3-VL-10B \
    --exp_name testset_Step3-VL-10B_think_0317 \
    --use_think \
    --work_dir ./
