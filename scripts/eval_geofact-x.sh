#!/bin/bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --gres=gpu:a100:4

#SBATCH -t 2:00:00
#SBATCH --constraint=24GB
#SBATCH --mem=128G

gpu_count=$(nvidia-smi -L | wc -l)

model_size=72
INPUT_FILE="data/geofakt_x_test.jsonl"
JUDGE_NAME="Qwen/Qwen2.5-${model_size}B-Instruct"

MODEL=""

RESULTS_FILE="results/GeoFactX_${MODEL}.jsonl"
MODEL_NAME="ckpts/${MODEL}"
ANALYSIS_PATH="geofact/${MODEL}"
OUTPUT_FILE="geofact/${MODEL}.jsonl"


python factaul_evaluation/GeoFact-X/llm_generation.py --test_file $INPUT_FILE --output_file $RESULTS_FILE --base_model $MODEL_NAME --use_vllm --tensor_parallel_size=$gpu_count --batch_size=128 --gpu_id=0,1,2,3


python factual_evaluation/GeoFact-X/llm_judge.py \
    --results_file=$RESULTS_FILE \
    --output_file=$OUTPUT_FILE \
    --analysis_output_dir=$ANALYSIS_PATH \
    --use_vllm \
    --batch_size 128 \
    --base_model=$JUDGE_NAME \
    --tensor_parallel_size=$gpu_count --gpu_id=0,1,2,3

