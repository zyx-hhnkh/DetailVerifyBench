#!/bin/bash
# 逐层注意力热力图可视化脚本
# 每层单独生成一张图片，文件名带 _L{layer:02d} 后缀

# ============ 通用配置 ============
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/mnt/sdc/model_zoo/Qwen3-VL-8B-Instruct"
MODEL_SELECT="Qwen3-VL-8B-Instruct"
IMAGE_DIR="./ms_data/test"

# ============ 幻觉检测可视化 (Normal Mode) ============
INPUT_FILE="./processing_data_zyx_260317/testset_gpt54_think_0326/inference_results/tested_model_output_seed_42_filtered.json"
OUTPUT_DIR="./processing_data_zyx_260317/testset_gpt54_think_0326/visualizations_perlayer"

python visualization/visualize_attention.py \
    --model_path ${MODEL_PATH} \
    --image_dir ${IMAGE_DIR} \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --model_select ${MODEL_SELECT} \
    --max_samples 1 \
    --batch_size 1 \
    --use_think
