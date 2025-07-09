import asyncio
import copy
import random
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
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from peft import LoraConfig
import transformers
import trl
from bridge_trainer import BridgeTrainer

from datasets import Value
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from googletrans import Translator
from utils.utils import count_japanese_no_latex, extract_boxed_answers, clean_latex

async def translate_text(text, dest='ja'):
    async with Translator() as translator:
        result = await translator.translate(text, dest)
    return result

async def detect_lang(text):
    async with Translator() as translator:
        result = await translator.detect(text)
        return result


def get_reward_funcs():

    from langid.langid import LanguageIdentifier, model
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    identifier.set_languages(['bn', 'de', 'es', 'en', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh'])
    def unicode_reward_func(prompts, completions, **kwargs):
        # Placeholder reward function that returns a constant value
        rewards = []
        for completion in completions:
            ja_count, clean_count = count_japanese_no_latex(completion)
            reward = ja_count / (clean_count + 1e-9)
            rewards.append(reward)
        print("Unicode Rewards:", rewards)
        return rewards

    def answer_reward_func(prompts, completions, **kwargs):
        targets = kwargs["answer"]
        rewards = []
        for completion, target in zip(completions, targets):
            pred = extract_boxed_answers(completion)
            if len(pred) > 0 and pred[-1] == target:
                rewards.append(1)
            else:
                rewards.append(0) 
        print("answer rewards", rewards)
        return rewards

    def langid_reward_func(prompts, completions, **kwargs):
        rewards = []
        prompts = [e.split('<|im_start|>user\n')[1].split('<|im_end|>')[0] for e in prompts]
        p_langs = []
        c_langs = []
        for prompt, completion in zip(prompts, completions):
            p_lang, p_conf = identifier.classify(clean_latex(prompt))
            c_lang, c_conf = identifier.classify(clean_latex(completion))
            p_langs.append(p_lang)
            c_langs.append(c_lang)
            if p_lang == c_lang:
                rewards.append(c_conf)
            else:
                rewards.append(0)
        print(rewards, p_langs, c_langs)
        return rewards

    def googletrans_reward_func(prompts, completions, **kwargs):
        rewards = []
        i = 0
        received = False

        prompts = [e.split('<|im_start|>user\n')[1].split('<|im_end|>')[0] for e in prompts]

        while not received and i < 10:
            try:
                prompt_langs = asyncio.run(detect_lang(prompts))
                completion_langs = asyncio.run(detect_lang([e[:5000] for e in completions])) # google translate length limit
                received = True
                break
            except:
                i += 1
        if not received or prompt_langs is None or completion_langs is None:
            return [0] * len(prompts)
        rewards = [cl.confidence if pl.lang == cl.lang else 0 for pl, cl in zip(prompt_langs, completion_langs)]
        pl_langs = [e.lang for e in prompt_langs]
        cl_langs = [e.lang for e in completion_langs]
        
        print(f"Google Translate Rewards: {rewards}, {pl_langs} {cl_langs}")
        return rewards
    return [answer_reward_func, langid_reward_func]

def add_prompt(dataset, langs):
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

def tokenize_prompt(dataset, tokenizer):
    """
        Apply tokenization to the prompt field in the dataset.
    """
    def _tokenize_prompt(example):
        prompt = example['prompt']
        messages_batch = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        tokenized = tokenizer.apply_chat_template(
                messages_batch,
                tokenize=False,
                add_generation_prompt=True
        )
        example['prompt'] = tokenized
        return example
    
    def _add_answer(example):
        answer = extract_boxed_answers(example['text'])
        example['answer'] = answer[-1] if len(answer) > 0 else None
        return example

    dataset['train'] = dataset['train'].map(_tokenize_prompt)
    dataset['train'] = dataset['train'].map(_add_answer)
    return dataset


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="LGSL")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    use_grpo : Optional[bool] = field(default=False)
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

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    try:
        num_nodes = dist.get_world_size()
    except:
        num_nodes = 1
    print(num_nodes)


    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    )

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
    if '.disk' in config.train_file_path:
        dataset = load_from_disk(config.train_file_path.replace('.disk', ''))
    elif 'science' in config.train_file_path:
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    
    if 'prompt' not in dataset['train'].features: # generate translated questions.
        langs = ['bn', 'de', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
        grpo_dataset = add_prompt(copy.deepcopy(dataset), langs)
        # save
        grpo_dataset.save_to_disk(os.path.join('./', 's1_random_prompt'))
        dataset = grpo_dataset
        print(args.output_dir)
    dataset = tokenize_prompt(dataset, tokenizer)

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

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    args.max_length = config.block_size
    grpo_config = get_grpo_config(config, args)
    grpo_config.max_completion_length = config.max_completion_length
    grpo_config.num_generations = 2
    grpo_config.max_steps = 10

    grpo_steps = []

    if config.use_grpo:
        for i in range(10000):
            if random.random() < config.grpo_sample_ratio:
                grpo_steps.append(i)

        trainer = BridgeTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            grpo_args=grpo_config,
            reward_funcs=get_reward_funcs(),
            data_collator=collator,
            grpo_sample_ratio=config.grpo_sample_ratio,
            grpo_steps=grpo_steps
        )
    trainer.train()

    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
#    torch.distributed.init_process_group(backend="nccl")
    train()
#    torch.distributed.destroy_process_group()
