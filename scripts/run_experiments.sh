#!/bin/bash
# 文件名: run_experiments.sh

# 接收参数
MODEL_NAME="$1"
MODEL_PATH="$2"
THINK_ARG="$3"   # 这里接收 "--use_think" 或者 "" (空字符串)
EXP_SUFFIX="$4"  # 例如 "think-20250113"

set -e # 报错即停

# 数据根目录
DATA_ROOT="/mnt/sdc/zhangyuxuan/Constructive-Decoding/results-V2"
# 基础实验名: Qwen3-VL-8B-think-20250113
BASE_EXP_NAME="${MODEL_NAME}-${EXP_SUFFIX}"

# 定义通用的参数前缀 (注意这里把 $THINK_ARG 放进去)
# 如果 THINK_ARG 为空，这里就不会有 --use_think
# 如果 THINK_ARG 为 --use_think，就会拼进去
COMMON_ARGS="--model_select ${MODEL_NAME} --model_path ${MODEL_PATH} --work_dir ./ ${THINK_ARG}"

echo ">> 开始执行实验序列... Exp Name Prefix: ${BASE_EXP_NAME}"

# 实验 0: Ours
echo "Running Exp 0: Ours..."
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --image_dir "./ms_data/test" \
    --input_json "./ms_data/test" \
    --exp_name "${BASE_EXP_NAME}-Ours" \
    ${COMMON_ARGS}

# 实验 1: Baseline
echo "Running Exp 1: Baseline..."
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --online \
    --image_dir "./ms_data/test" \
    --input_json \
        "${DATA_ROOT}/ChartX_test300/no_pos_inject_temp0_2.jsonl" \
        "${DATA_ROOT}/DOCCI_test300/no_pos_inject_temp0_2.jsonl" \
    --exp_name "${BASE_EXP_NAME}-Baseline" \
    ${COMMON_ARGS}

# 实验 2: PI
echo "Running Exp 2: PI (Original)..."
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --online \
    --image_dir "./ms_data/test" \
    --input_json \
        "${DATA_ROOT}/ChartX_test300/ORIGINAL-clause_threshold5_0_top6_temp0_2.jsonl" \
        "${DATA_ROOT}/DOCCI_test300/ORIGINAL-clause_threshold5_0_top6_temp0_2.jsonl" \
    --exp_name "${BASE_EXP_NAME}-Original" \
    ${COMMON_ARGS}

# 实验 3: Halsum
echo "Running Exp 3: Halsum..."
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --online \
    --image_dir "./ms_data/test" \
    --input_json \
        "${DATA_ROOT}/ChartX_test300/clause_threshold5_0_halsum450_temp0_2.jsonl" \
        "${DATA_ROOT}/DOCCI_test300/clause_threshold5_0_halsum532_temp0_2.jsonl" \
    --exp_name "${BASE_EXP_NAME}-Halsum" \
    ${COMMON_ARGS}

echo ""
echo "✅ 所有实验运行完毕！"