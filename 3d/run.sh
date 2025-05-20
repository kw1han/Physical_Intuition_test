#!/bin/bash

# 设置数据和输出目录位置
DATA_ROOT="/home/student0/Physical_Intuition_test/3d/Phytion_new"
OUTPUT_DIR="/home/student0/Physical_Intuition_test/3d/result_3d"
NUM_VIDEOS=150  # 测试视频数量
SHOTS="0 1 2"  # Shot设置
GAME_TYPE="Collide"  # 要测试的游戏类型，可选值：Collide, Contain, Dominoes, Drape, Drop, Link, Roll, Support
MODEL_NAME="deepseek-vl2"  # 模型名称
API_KEY="sk-uruhfkcrzkebvehlyoeradzzwentjyjpqbteiryddkkpahpe"  # API密钥
BASE_URL="https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions"  # API基础URL

echo "开始三维物理直觉测试..."
echo "数据目录: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "测试游戏类型: $GAME_TYPE"
echo "使用模型: $MODEL_NAME"
echo "将测试 $NUM_VIDEOS 个视频，使用不同的shot设置: $SHOTS"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 运行测试
echo "===== 开始测试 ====="
python3 3D_test2.py \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --num_videos "$NUM_VIDEOS" \
  --shots $SHOTS \
  --game_type "$GAME_TYPE" \
  --model_name "$MODEL_NAME" \
  --api_key "$API_KEY" \
  --base_url "$BASE_URL"

echo "测试完成！"
echo "结果保存在: $OUTPUT_DIR"