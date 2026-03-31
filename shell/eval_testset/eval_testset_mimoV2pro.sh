cd "$(dirname "$0")/../.."

JSONL_DIR=./ms_data/test/test_gt_withtag.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select mimoV2pro \
    --exp_name testset_mimoV2pro_think \
    --use_think \
    --work_dir ./
