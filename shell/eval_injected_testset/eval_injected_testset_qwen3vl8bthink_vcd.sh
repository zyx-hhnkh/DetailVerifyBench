cd "$(dirname "$0")/../.."

MODEL_DIR=/mnt/sdc/model_zoo/Qwen3-VL-8B-Thinking
JSONL_DIR=/mnt/sdc/Hallucination_DATA/Hallucination_Bench/testset_geminiflash_vs_gpt52_n2.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi \
    --plugin VCD \
    --model_path $MODEL_DIR \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select Qwen3-VL-8B-Thinking \
    --exp_name injected_testset_Qwen3-VL-8B_think_VCD \
    --work_dir ./ \
    --vcd_gamma 0.3 \
    --vcd_threshold -2.0 
