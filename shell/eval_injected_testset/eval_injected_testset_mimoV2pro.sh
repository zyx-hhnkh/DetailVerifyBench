cd "$(dirname "$0")/../.."

JSONL_DIR=PATH_TO_SYNTHETIC_JSONL  # path to synthetic hallucination test jsonl
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select mimoV2pro \
    --exp_name injected_testset_mimoV2pro_think \
    --use_think \
    --work_dir ./
