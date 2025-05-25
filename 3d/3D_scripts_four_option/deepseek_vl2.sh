SCENES=("Contain" "Drop" "Link" "Roll" "Support" "Drape" "Dominoes")

BASE_PARAMS=" --model_name deepseek-ai/deepseek-vl2 \
  --api_key sk-uruhfkcrzkebvehlyoeradzzwentjyjpqbteiryddkkpahpe \
  --base_url https://api.siliconflow.cn/v1 \
  --data_root /home/student0/Physical_Intuition_test/3d/Physion/Physion_image \
  --num_sets 75 \
  --repetitions 4 \
  --max_tokens 512 \
  --temperature 0.7 \
  --top_p 0.7 \
  --stream False \
  --frequency_penalty 0.5 \
  --n 1 "
for scene in "${SCENES[@]}"; do
    echo "开始处理场景: $scene"
    # 创建输出目录
    OUTPUT_DIR="/home/student0/Physical_Intuition_test/3d/results_3d/four_option/deepseek-vl2/${scene}"
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

echo "所有场景处理完成！"

