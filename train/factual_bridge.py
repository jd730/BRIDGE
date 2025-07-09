import os
import json
import sys
import asyncio
import random
sys.path.append('..')
sys.path.append('.')
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, DatasetDict, load_from_disk
import transformers
import trl
from googletrans import Translator
from bridge_trainer import BridgeTrainer

from reward_functions import language_matching_reward_func_advanced, format_reward_func_factual

async def translate_text(text, dest='ja'):
    async with Translator() as translator:
        result = await translator.translate(text, dest)
    return result

async def detect_lang(text):
    async with Translator() as translator:
        result = await translator.detect(text)
        return result


def get_reward_funcs():
    return [language_matching_reward_func_advanced, format_reward_func_factual]

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="LGSL")
    train_file_path: Optional[str] = field(default='data/geofakt_x_train_sft.jsonl')
    grpo_sample_ratio: Optional[float] = field(default=1)
    max_completion_length: Optional[int] = field(default=1024)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def get_grpo_config(config, args):
    """
        Create a GRPO configuration from the provided arguments.
    """

    grpo_config = trl.GRPOConfig()
    # check all arguments in grpo and if it is also included in config or args then set item in get_grpo_config
    for key, value in vars(grpo_config).items():
        if hasattr(config, key):
            setattr(grpo_config, key, getattr(config, key))
        elif hasattr(args, key):
            setattr(grpo_config, key, getattr(args, key))
    return grpo_config


def tokenize_prompt(dataset, tokenizer, langs=[], no_step_format=True):
    """
        Apply tokenization to the prompt field in the dataset.
    """
    def _tokenize_prompt(example):
        question = example['question']
        if len(langs) > 0:
            dest = random.sample(langs, 1)[0]
            try:
                question = asyncio.run(translate_text(question, dest)).text
                print(question)
            except:
                print("Translation Fail")
                pass
        reasoning = example['reasoning']
        answer = example.get('answer_list')[0]
        if no_step_format and False: # and False:
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
                {"role": "assistant", "content": f"Let me think step by step.\n<think>{reasoning}</think><answer>{answer}</answer>"} # 250624 fixed. Add </think>
                )
        
        if 'translated_question' in example:
            translated_question = example['translated_question']
            r1_prefix[-2] = {"role": "user", "content": f"{translated_question} Please think carefully and return your reasoning inside <think> </think> tags, and the final direct answer inside <answer> </answer> tags."}

        tokenized = tokenizer.apply_chat_template(
                r1_prefix,
                tokenize=False,
#                add_generation_prompt=True
        )
        tokenized_text = tokenizer.apply_chat_template(
                sft_prefix,
                tokenize=False,
                continue_final_message=False,
        )

        example['prompt'] = tokenized
        example['text'] = tokenized_text
        example['answer'] = answer
        return example
    

    dataset['train'] = dataset['train'].map(_tokenize_prompt)
    return dataset

def translate_question(dataset, langs):
    """
        Add prompt for GRPO.
    """
    def _add_prompt(example):
        dest = random.sample(langs, 1)[0]
        try:
            example['prompt'] = asyncio.run(translate_text(example['question'], dest)).text
        except:
            print("Translation Fail")
            example['prompt'] = example['question']
        return example
    dataset['train'] = dataset['train'].map(_add_prompt)
    return dataset

def change_argname(dataset):
    def _modification(example):
        example['translated_question'] = example.pop('prompt')
        return example
    dataset['train'] = dataset['train'].map(_modification)
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


    
    translated_path = config.train_file_path.replace('.jsonl', '_bridge_translated')
    if not os.path.exists(translated_path):
        print("Create Dataset")
        dataset = load_dataset("json", data_files=config.train_file_path)
        langs = ['en', 'zh', 'fr', 'ja', 'hi', 'ru', 'ar', 'el', 'ne', 'uk', 'tr', 'sw', 'th']
#        langs = ['zh', 'fr', 'ru', 'ar', 'el', 'ne', 'uk', 'tr']
        dataset = translate_question(dataset, langs)
        print("Translation is Done", langs, dataset)
        dataset.save_to_disk(translated_path)
    elif '.json' in config.train_file_path:
        dataset = load_dataset("json", data_files=config.train_file_path)
    else:
        dataset = load_from_disk(config.train_file_path)
        dataset = change_argname(dataset)
#        output_path = 'temp.jsonl'
#        with open(output_path, "w", encoding="utf-8") as f:
#            for datum in dataset['train']:
#                out = json.dumps(datum)
#                f.write(out + '\n')
#        breakpoint()

    langs = []
    dataset = tokenize_prompt(dataset, tokenizer, langs=langs)

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

    args.remove_unused_columns = False

    # GRPO part.
    grpo_config = get_grpo_config(config, args)
    grpo_config.max_completion_length = config.max_completion_length
    grpo_config.num_generations = 8
    grpo_config.max_steps = 2


    trainer = BridgeTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            grpo_args=grpo_config,
            reward_funcs=get_reward_funcs(),
            data_collator=collator,
            grpo_sample_ratio=config.grpo_sample_ratio,
        )
    trainer.train()


    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
#    torch.distributed.init_process_group(backend="nccl")
    train()
#    torch.distributed.destroy_process_group()
