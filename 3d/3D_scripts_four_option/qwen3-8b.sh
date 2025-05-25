SCENES=("Contain" "Drop" "Link" "Roll" "Support" "Drape" "Dominoes")

BASE_PARAMS=" --model_name qwen3-8b \
  --api_key sk-195225bba1f44e37aa394f1841d86a8e \
  --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --data_root /home/student0/Physical_Intuition_test/3d/Physion/Physion_image \
  --num_sets 1 \
  --repetitions 4 \
  --max_tokens 512 \
  --temperature 0.7 \
  --top_p 0.8 \
  --stream True \
  --presence_penalty 1.5 "
for scene in "${SCENES[@]}"; do
    echo "开始处理场景: $scene"
    # 创建输出目录
    OUTPUT_DIR="/home/student0/Physical_Intuition_test/3d/results_3d/four_option/qwen3-8b/${scene}"
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