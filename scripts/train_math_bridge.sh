#!/bin/bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=128G
#SBATCH -t 10:00:00

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500  # Change if needed
export NNODES=$SLURM_NNODES
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
export NODE_RANK=$SLURM_NODEID

echo "Master Node: $MASTER_ADDR"
echo "Running on $WORLD_SIZE GPUs across $NNODES nodes"

uid="$(date +%Y%m%d_%H%M%S)"
model_size=7
base_model="Qwen/Qwen2.5-${model_size}B-Instruct"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false
grpo_sample_ratio=0.01

output_dir="ckpts/MathBRIDGE_${model_size}B-${grpo_sample_ratio}-E${epochs}-${uid}"

srun accelerate launch \
     --config_file deepspeed_zero3.yaml \
     --num_processes $WORLD_SIZE \
     --num_machines $NNODES \
     --main_process_ip $MASTER_ADDR \
     --machine_rank $NODE_RANK --main_process_port $MASTER_PORT \
     --rdzv_backend c10d \
    train/math_bridge.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir=${output_dir} \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --use-liger-kernel \
    --grpo_sample_ratio=${grpo_sample_ratio} --use_grpo

