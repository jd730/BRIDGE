model_size=72
INPUT_FILE="../../data/geofakt_x_test.jsonl"
RESULTS_FILE="generations_qwen25-7b-Instruct.jsonl"
OUTPUT_FILE="reasoning_results_vllm_qwen25-7b-Instruct.jsonl"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
FILE_NAME="Qwen2.5-7B-Instruct"
ANALYSIS_PATH="analysis_results/$FILE_NAME"
JUDGE_NAME="Qwen/Qwen2.5-${model_size}"B-Instruct"

python llm_generation.py \
    --test_file $INPUT_FILE \
    --output_file $RESULTS_FILE \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --use_vllm \

python llm_judge.py \
    --results_file=$RESULTS_FILE \
    --output_file=$OUTPUT_FILE \
    --analysis_output_dir=$ANALYSIS_PATH \
    --use_vllm \
    --base_model=$JUDGE_NAME
