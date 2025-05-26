#!/bin/bash

# 测试qwen3-8b模型
# 用法: ./test_qwen3_8b.sh [game_type]
# 示例:
#   测试所有游戏类型: ./test_qwen3_8b.sh
#   测试特定游戏类型: ./test_qwen3_8b.sh Falling

GAME_TYPE=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROJECT_ROOT="/home/student0/Physical_Intuition_test"
OUTPUT_ROOT="$PROJECT_ROOT/test_results/prediction_accuracy"

# 创建输出目录
OUTPUT_DIR="$OUTPUT_ROOT/qwen3_8b_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# 设置日志文件
LOG_FILE="$OUTPUT_DIR/experiment.log"
exec 1> >(tee -a "$LOG_FILE") 2>&1  # 将所有输出重定向到日志文件，同时保持终端输出

echo "========================================"
echo "开始测试 qwen3-8b 模型"
echo "测试时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
if [ ! -z "$GAME_TYPE" ]; then
    echo "测试游戏类型: $GAME_TYPE"
fi
echo "========================================"

# 运行测试
python ../test_prediction_accuracy/test_ollama.py \
    --data_root "$PROJECT_ROOT/balanced_dataset" \
    --model_name "qwen3-8b" \
    --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api_key "sk-195225bba1f44e37aa394f1841d86a8e" \
    --output_dir "$OUTPUT_DIR" \
    --prompt_dir "$PROJECT_ROOT/prompt1" \
    --num_trials 52 \
    --temperature 0.0 \
    --seed 42 \
    ${GAME_TYPE:+--game_type "$GAME_TYPE"}

echo ""
echo "========================================"
echo "测试完成！"
echo "结束时间: $(date)"
echo "结果保存在: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "======================================== 