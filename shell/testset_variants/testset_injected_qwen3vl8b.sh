cd "$(dirname "$0")/../.."

JSONL_DIR1=PATH_TO_SYNTHETIC_JSONL_1  # path to synthetic jsonl (n=1)
JSONL_DIR2=PATH_TO_SYNTHETIC_JSONL_2  # path to synthetic jsonl (n=2)
JSONL_DIR3=PATH_TO_SYNTHETIC_JSONL_3  # path to synthetic jsonl (n=3)
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory
MODEL_DIR=PATH_TO_MODEL          # path to local model directory

CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --mode advi \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR1 \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name testset0318_injected_n1_qwen3vl8b_think \
    --use_think \
    --work_dir ./

CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --mode advi \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR2 \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name testset0318_injected_n2_qwen3vl8b_think \
    --use_think \
    --work_dir ./

CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --mode advi \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR3 \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name testset0318_injected_n3_qwen3vl8b_think \
    --use_think \
    --work_dir ./
