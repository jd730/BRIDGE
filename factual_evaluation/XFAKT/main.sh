# Base Model Evaluation

LLM Generation on Factual dataset
python llm_generation.py --model_name "Qwen2.5-7B-Instruct-base" \
        --model "Qwen/Qwen2.5-7B-Instruct" \
        --batch_size 8 \
        --gpu_ids "0" \
        --max_tokens 256

# Using another LLM as a judge for above genrations comparisions
python llm_judge.py \
    --model "Qwen2.5-7B-Instruct-base" \
    --evaluator_model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "factual_recall" \
    --prompt "without_system_prompt" \
    --batch_size 8 \
    --gpu_ids "0" \
    --max_tokens 256

# Score calculation
BASE_DIR="./results/factual_recall"
# List of models to evaluate
MODELS="Qwen2.5-7B-Instruct-base"

# Output directory for metrics
OUTPUT_DIR="metrics_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Construct path to results for this model
RESULTS_PATH="$BASE_DIR/Qwen2.5-7B-Instruct-base/without_system_prompt"

python scores.py \
    --results_dir "$RESULTS_PATH" \
    --model_name "$MODELS" \
    --output_dir "$OUTPUT_DIR"
 
echo "All metrics calculation complete!"

