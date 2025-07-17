<p align="center">
  <a href="https://arxiv.org/abs/2507.05418">
    <img src="https://img.shields.io/badge/arXiv-2507.05418-b31b1b.svg" alt="arXiv preprint">
  </a>
</p>

<div align="center">
  <h1>Learn Globally, Speak Locally:<br>Bridging the Gaps in Multilingual Reasoning</h1>
</div>

**TL;DR:** We introduce GeoFact-X and BRIDGE to evaluate and improve multilingual reasoning in LLMs by aligning internal reasoning with the input language using language-consistency rewards.


## Repository Structure

- `eval/`: Evaluation tools for mathematical reasoning
- `dataset/`: GeoFact-X dataset
- `factual_evaluation/`: Factual reasoning evaluation scripts (GeoFact-X and X-FaKT)
- `data/`: Synthetic data generation scripts
- `scripts/`: Shell scripts for training and evaluation
- `train/`: Python training scripts
- `utils/`: Utility functions and helpers


## Training

Use the scripts in `scripts/` to launch training.

**Hardware recommendations:**
- Factual reasoning: ≥ 4 A100 GPUs  
- Mathematical reasoning: ≥ 8 A100 GPUs  

### Example: Multi-node training with Slurm

```bash
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
epochs=5
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
push_to_hub=false
grpo_sample_ratio=0.01

srun accelerate launch \
  --config_file deepspeed_zero3.yaml \
  --num_processes $WORLD_SIZE \
  --num_machines $NNODES \
  --main_process_ip $MASTER_ADDR \
  --machine_rank $NODE_RANK \
  --main_process_port $MASTER_PORT \
  --rdzv_backend c10d \
  train/bridge.py \
    --grpo_sample_ratio=${grpo_sample_ratio} --use_grpo \
    --block_size=20000 \
    --train_file_path="simplescaling/s1K-1.1_tokenized" \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --model_name=${base_model} \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 --adam_beta2=0.95 \
    --output_dir="ckpts/bridge_${model_size}b-${grpo_sample_ratio}-ep${epochs}-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --use-liger-kernel \
    --gradient_checkpointing=True
```

# Evaluation 
## Factual Reasoning
Update model paths in the scripts as needed:
```
sh scripts/eval_xfakt.sh
sh scripts/eval_geofact-x.sh
```

## Mathematical Reasoning
We cloned [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at commit `4cec66e4e468d15789473d6d63c3a61a751fa524` and modified it. Setup:
```bash
cd eval/lm-evaluation-harness
pip install -e .[math,vllm]
```

If you want to compute statistics (avg thinking tokens etc) for an evaluation run you can use 
`python eval/compute_sample_stats.py path_to_samples_file.jsonl`

All our evaluation result files are at: https://hf.co/datasets/simplescaling/results

To run REBASE: commands are in `eval/rebase/run.sh`


## MGSM and MMLU-ProX Math for fast evaluation

```bash
tasks='mgsm_native_cot_bn,mgsm_native_cot_de,mgsm_native_cot_es,mgsm_native_cot_fr,mgsm_native_cot_ru,mgsm_native_cot_sw,mgsm_native_cot_te,mgsm_native_cot_th,mgsm_native_cot_zh,mgsm_native_cot_en,mgsm_native_cot_ja,mmlu_prox_ar_math,mmlu_prox_bn_math,mmlu_prox_de_math,mmlu_prox_es_math,mmlu_prox_fr_math,mmlu_prox_en_math,mmlu_prox_hi_math,mmlu_prox_ja_math,mmlu_prox_ko_math,mmlu_prox_pt_math,mmlu_prox_sw_math,mmlu_prox_th_math,mmlu_prox_zh_math'
lm_eval --model vllm --model_args pretrained=${model_name},dtype=auto,tensor_parallel_size=${num_gpu},gpu_memory_utilization=0.90,max_model_len=20000 --tasks $tasks --batch_size auto --apply_chat_template --output_path ${output_dir} --log_samples --gen_kwargs "max_gen_toks=20000"
```

## Measure Language Performance
```bash
python3 utils/language_detector.py
```

## Acknowledgement
This codebase is based on https://github.com/simplescaling/s1.



## Citation
If you use this code for your research, please cite our paper.


```
@article{hwang2025learn,
      title={Learn Globally, Speak Locally: Bridging the Gaps in Multilingual Reasoning},
      author={Hwang, Jaedong and Tanmay, Kumar and Lee, Seok-Jin and Agrawal, Ayush and Palangi, Hamid and Ayush, Kumar and Fiete, Ila R and Liang, Paul Pu},
      journal={arXiv preprint arXiv:2507.05418},
      year={2025}
    }
```
