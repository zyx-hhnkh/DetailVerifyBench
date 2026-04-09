cd "$(dirname "$0")/../.."

JSONL_DIR1=PATH_TO_SYNTHETIC_JSONL_1  # path to synthetic jsonl (n=1)
JSONL_DIR2=PATH_TO_SYNTHETIC_JSONL_2  # path to synthetic jsonl (n=2)
JSONL_DIR3=PATH_TO_SYNTHETIC_JSONL_3  # path to synthetic jsonl (n=3)
IMAGE_DIR=PATH_TO_IMAGE_DIR        # path to test image directory

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi_nodetect \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR1 \
    --model_select gpt-5.2 \
    --exp_name testset0318_injected_nodetect_n1_gpt52_think \
    --use_think \
    --work_dir ./
