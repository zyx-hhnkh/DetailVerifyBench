cd "$(dirname "$0")/../.."

MODEL_DIR=PATH_TO_MODEL          # path to local model directory
JSONL_DIR=PATH_TO_SYNTHETIC_JSONL  # path to synthetic hallucination test jsonl
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select Step3-VL-10B \
    --exp_name injected_testset_Step3-VL-10B_think \
    --use_think \
    --work_dir ./
