cd "$(dirname "$0")/../.."

MODEL_DIR=PATH_TO_MODEL          # path to local model directory
JSONL_DIR=PATH_TO_REAL_JSONL       # path to real hallucination test jsonl
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory

CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --mode offline \
    --plugin VCD \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name testset_Qwen3-VL-8B_think_VCD \
    --work_dir ./ \
    --vcd_gamma 0.3 \
    --vcd_threshold -2.0
