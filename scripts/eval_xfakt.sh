#!/bin/bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --gres=gpu:a100:4

#SBATCH -t 4:00:00
#SBATCH --constraint=24GB
#SBATCH --mem=128G

gpu_count=$(nvidia-smi -L | wc -l)
batch_size=128
max_tokens=1024

# List of models to evaluate
BASE_DIR="./XFakT/factual_recall"
# Create output directory
# Output directory for metrics

model_name=""
MODEL="ckpts/${model_name}"
RESULTS_PATH="$BASE_DIR/${model_name}/without_system_prompt"
# Construct path to results for this model
python factual_evaluation/XFAKT/llm_generation.py --max_tokens $max_tokens --model $MODEL --tensor_parallel_size=$gpu_count --model_name $model_name --batch_size $batch_size --gpu_ids="0,1,2,3"
python factual_evaluation/XFAKT/llm_judge.py --model $model_name --evaluator_model "Qwen/Qwen2.5-72B-Instruct"  --dataset "factual_recall" --prompt "without_system_prompt" --batch_size $batch_size --gpu_ids="0,1,2,3" --max_tokens 256 --tensor_parallel_size=$gpu_count
python factual_evaluation/XFAKT/scores.py --results_dir $RESULTS_PATH --model_name $model_name --output_dir XFakT/summary
 
echo "All metrics calculation complete!"

