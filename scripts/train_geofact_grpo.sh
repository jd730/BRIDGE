#!/bin/bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=24GB
#SBATCH --mem=128G
#SBATCH -t 10:00:00

export VAR=`bc -l <<< "${SLURM_ARRAY_TASK_ID}"`

echo $VAR

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=22900  # Change if needed
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
export NODE_RANK=$SLURM_NODEID

echo "Master Node: $MASTER_ADDR"

echo "Running on $WORLD_SIZE GPUs across $NNODES nodes"

## Reference Running: bash train/sft.sh
uid="$(date +%Y%m%d_%H%M%S)"

model_size=7

base_model="Qwen/Qwen2.5-${model_size}B-Instruct"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=32 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

output_dir="ckpts/GeoFactGRPO_${grpo_sample_ratio}-${model_size}B_E${epochs}B${micro_batch_size}"
gpu_count=$(nvidia-smi -L | wc -l)

train_file_path="dataset/geofact_x_train_grpo.jsonl" 


accelerate launch --config_file deepspeed_zero3.yaml --num_processes $gpu_count --num_machines 1 --machine_rank 0 --main_process_port $MASTER_PORT --rdzv_backend c10d  train/geofact_grpo.py \
    --max_completion_length=1024 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9  --adam_beta2=0.95 \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --output_dir=${output_dir} \
    --gradient_checkpointing=True \
    --use-liger-loss=True \
    --use-liger-kernel=True \
    --num_generations=8 \



