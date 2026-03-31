cd "$(dirname "$0")/../.."

JSONL_DIR=./ms_data/test/test_gt_withtag.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select gemini-3-pro \
    --exp_name testset_gemini3pro_think_0317 \
    --use_think \
    --work_dir ./ 

