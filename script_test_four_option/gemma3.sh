SCENES=("SeeSaw" "Shafts" "Table" "Unbox" "Unsupport")

BASE_PARAMS="--model_name gemma3:27b \
  --api_key ollama \
  --base_url http://localhost:11434/v1 \
  --data_root /home/student0/Physical_Intuition_test/balanced_dataset \
  --num_sets 1 \
  --repetitions 4 \
  --max_tokens 8192 \
  --temperature 1.0 \
  --top_p 0.95 \
  --stream False \
  --frequency_penalty 0.5 \
  --n 1 "
for scene in "${SCENES[@]}"; do
    echo "开始处理场景: $scene"
    # 创建输出目录
    OUTPUT_DIR="/home/student0/Physical_Intuition_test/test_results/four_option/gemma3/${scene}"
    mkdir -p "$OUTPUT_DIR"
    
    # 运行测试
    python3 /home/student0/Physical_Intuition_test/test_four_option_eval/test_unified.py \
        $BASE_PARAMS \
        --output_dir "$OUTPUT_DIR" \
        --game_type "$scene"
    
    echo "完成场景: $scene"
    echo "------------------------"
    
    # 等待一段时间，避免可能的资源冲突
    sleep 5
done