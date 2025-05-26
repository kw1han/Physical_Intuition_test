SCENES=("Dominoes")

BASE_PARAMS=" --model_name gpt-4o \
  --api_key sk-DXv1XQJr7HxWOZaVskIMYC0b7M8S9MAmlkjFMmffejDQ09GH \
  --base_url https://xiaoai.plus/v1 \
  --data_root /home/student0/Physical_Intuition_test/3d/Physion/Physion_image \
  --num_sets 1 \
  --repetitions 4 \
  --max_tokens 512 \
  --temperature 0.7 \
  --top_p 0.9 "
for scene in "${SCENES[@]}"; do
    echo "开始处理场景: $scene"
    # 创建输出目录
    OUTPUT_DIR="/home/student0/Physical_Intuition_test/3d/results_3d/four_option/gpt_4o/${scene}"
    mkdir -p "$OUTPUT_DIR"
    
    # 运行测试
    python /home/student0/Physical_Intuition_test/3d/3D_test_four_option.py \
        $BASE_PARAMS \
        --output_dir "$OUTPUT_DIR" \
        --game_type "$scene"
    
    echo "完成场景: $scene"
    echo "------------------------"
    
    # 等待一段时间，避免可能的资源冲突
    sleep 5
done