#!/bin/bash

# 所有要测试的 game type
GAME_TYPES=("Basic" "Bridge" "Catapult" "Chaining" "Falling" "Gap" "Launch" "Prevention" "SeeSaw" "Shafts" "Table" "Unbox" "Unsupport")

# 公共参数
MODEL_NAME="gpt-4o"
API_KEY="sk-DXv1XQJr7HxWOZaVskIMYC0b7M8S9MAmlkjFMmffejDQ09GH"
BASE_URL="https://xiaoai.plus/v1"
DATA_ROOT="/home/student0/Physical_Intuition_test/balanced_dataset"
RESULTS_ROOT="/home/student0/Physical_Intuition_test/test_results/four_option/gpt_4o"
NUM_SETS=10
REPETITIONS=4
MAX_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9

for GAME_TYPE in "${GAME_TYPES[@]}"
do
  OUTDIR="${RESULTS_ROOT}/${GAME_TYPE}"
  mkdir -p "$OUTDIR"
  echo "==== Running for game_type: $GAME_TYPE ===="
  python3 ../test_four_option_eval/test_unified.py \
    --model_name "$MODEL_NAME" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTDIR" \
    --num_sets "$NUM_SETS" \
    --repetitions "$REPETITIONS" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --game_type "$GAME_TYPE"
done