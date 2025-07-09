import os
import sys
sys.path.append('..')
sys.path.append('.')
import pickle
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import LoraConfig
import transformers
import trl


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="LGSL")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.GKDConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")


    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    
    if 'science' in config.train_file_path:
        dataset = load_dataset('simplescaling/s1K-1.1_tokenized')
        dataset['train'] = dataset['train'].filter(lambda example: example['cot_type'] == 'science')
        config.train_file_path = 'simplescaling/s1K-1.1_tokenized'
    elif 'math' in config.train_file_path:
        dataset = load_dataset('simplescaling/s1K-1.1_tokenized')
        dataset['train'] = dataset['train'].filter(lambda example: example['cot_type'] == 'math')
        config.train_file_path = 'simplescaling/s1K-1.1_tokenized'
    elif 'jsonl' in config.train_file_path:
        dataset = load_dataset("json", data_files=config.train_file_path)
    else:
        dataset = load_dataset(config.train_file_path)
    print(dataset)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name or "qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    args.max_length = config.block_size


    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )
    
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
#    torch.distributed.init_process_group(backend="nccl")
    train()
#    torch.distributed.destroy_process_group()
