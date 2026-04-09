cd "$(dirname "$0")/../.."

JSONL_DIR=PATH_TO_REAL_JSONL       # path to real hallucination test jsonl
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select gemini-3-pro \
    --exp_name testset_gemini3pro_think \
    --use_think \
    --work_dir ./

