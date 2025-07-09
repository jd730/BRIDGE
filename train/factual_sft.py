import os
import sys
sys.path.append('..')
sys.path.append('.')
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
import transformers
import trl
from liger_kernel.transformers import AutoLigerKernelForCausalLM


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="LGSL")
    train_file_path: Optional[str] = field(default='dataset/geofact_x_train_sft.jsonl')

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project


def tokenize_prompt(dataset, tokenizer, no_step_format=True):
    """
        Apply tokenization to the prompt field in the dataset.
    """
    def _tokenize_prompt(example):
        question = example['question']
        reasoning = example['reasoning']
        answer = example.get('answer_list')[0]
        if no_step_format and False :
            reasoning.replace('<step>', '')
        r1_prefix = [
            {"role": "system", "content": "You are a helpful assistant. When answering a factual question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Think and reason carefully in the same language as the question (for example, if the question is in Hindi, then think and reason in Hindi).\n3. Consider multiple perspectives and potential answers before settling on your final response.\n4. Evaluate the confidence in your answer based on the information available to you.\n5. Provide the final answer clearly in the same language as the question, making sure it's well-supported by your reasoning.\n6. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."
            },
            {"role": "system", "content": "You are a helpful assistant. When answering a factual question, first think and reason in the same language as the question (for example, if question is in Hindi then think and reason in Hindi). Then, provide the final answer clearly in that same language."},
            {"role": "user", "content": f"{question} Please think carefully and return your reasoning inside <think> </think> tags, and the final direct answer inside <answer> </answer> tags."},
            {"role": "assistant", "content": "Let me think step by step.\n<think>"}
        ]
        sft_prefix = r1_prefix[:-1]
        sft_prefix.append(
                {"role": "assistant", "content": f"Let me think step by step.\n<think>{reasoning}</think><answer>{answer}</answer>"}
                )
        
        tokenized_text = tokenizer.apply_chat_template(
                sft_prefix,
                tokenize=False,
                continue_final_message=False,
        )
        example['text'] = tokenized_text
        return example
    

    dataset['train'] = dataset['train'].map(_tokenize_prompt)
    return dataset



def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    # setting up trainer
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


    dataset = load_dataset("json", data_files=config.train_file_path)
    dataset = tokenize_prompt(dataset, tokenizer)
    print(dataset)

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
