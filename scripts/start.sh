#!/bin/bash
# 文件名: start.sh

# ================= 输入参数 =================
# 1. 模型名称 (例如: Qwen3-VL-8B-Thinking)
INPUT_MODEL_NAME="$1"
# 2. 是否Think (true/false)
IS_THINK="$2"

# ================= 检查参数 =================
if [ -z "$INPUT_MODEL_NAME" ] || [ -z "$IS_THINK" ]; then
    echo "❌ 用法错误！"
    echo "👉 格式: bash start.sh <模型名称> <是否Think(true/false)>"
    echo "👉 示例(开启): bash start.sh Qwen3-VL-8B-Thinking true"
    echo "👉 示例(关闭): bash start.sh Qwen3-VL-8B-Thinking false"
    exit 1
fi

# ================= 路径配置 =================
# 请确保这里的模型根目录是正确的
MODEL_BASE_DIR=PATH_TO_MODEL_ZOO  # base directory containing all model directories
FULL_MODEL_PATH="${MODEL_BASE_DIR}/${INPUT_MODEL_NAME}"

if [ ! -d "$FULL_MODEL_PATH" ]; then
    echo "❌ 错误: 找不到模型路径 -> $FULL_MODEL_PATH"
    exit 1
fi

# ================= 逻辑处理 =================
CURRENT_DATE=$(date +%Y%m%d)

# 初始化变量
CMD_ARG=""   # 传给python的参数
MODE_TAG=""  # 既然用于文件名

if [ "$IS_THINK" == "true" ]; then
    # 开启模式: 传入参数，标记为 think
    CMD_ARG="--use_think"
    MODE_TAG="think"
else
    # 关闭模式: 参数留空，标记为 nothink
    CMD_ARG=""
    MODE_TAG="nothink"
fi

# 构建实验名后缀: think-20250113 或 nothink-20250113
EXP_SUFFIX="${MODE_TAG}-${CURRENT_DATE}"

echo "============================================"
echo "🚀 任务启动确认"
echo "--------------------------------------------"
echo "🤖 模型: $INPUT_MODEL_NAME"
echo "📅 日期: $CURRENT_DATE"
echo "🧠 模式: $MODE_TAG"
echo "🔧 参数: ${CMD_ARG:-"(无参数/NoThink)"}"
echo "============================================"

# ================= 调用执行脚本 =================
# 参数顺序: 1.模型名 2.模型路径 3.Think参数(可能为空) 4.实验后缀
# 注意: "$CMD_ARG" 需要带引号以防止空字符串被忽略位置，但在接收端我们会处理它
bash ./scripts/run_experiments.sh "$INPUT_MODEL_NAME" "$FULL_MODEL_PATH" "$CMD_ARG" "$EXP_SUFFIX"
