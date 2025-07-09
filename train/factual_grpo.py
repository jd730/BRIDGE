import logging
import os
from dataclasses import dataclass
from datetime import datetime
import re
import random
import torch
from datasets import Dataset, load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
# Set up SummaryWriter for more detailed logging
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
import numpy as np
from typing import List, Optional, Tuple, Any
from reward_functions import correctness_reward_func_factual, correctness_reward_func_factual_fuzzy, language_matching_reward_func_advanced, format_reward_func_factual

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = None
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)



############
class DetailedTensorboardCallback(TrainerCallback):
    """Custom callback for logging more detailed metrics to TensorBoard."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Event called after logging the last logs."""
        if not hasattr(args, "writer") or args.writer is None:
            return
            
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                # Log with high precision (7 decimal places)
                args.writer.add_scalar(key, float(f"{value:.7f}"), state.global_step)
                
                # Special handling for loss values - log to console with higher precision
                if "loss" in key.lower():
                    print(f"Step {state.global_step}: {key} = {value:.7f}")
                    
        # Additional useful metrics that might not be in logs
        if hasattr(state, "train_metrics") and state.train_metrics:
            for reward_key, reward_val in state.train_metrics.items():
                if "reward" in reward_key and isinstance(reward_val, (int, float)):
                    args.writer.add_scalar(f"rewards/{reward_key}", float(f"{reward_val:.7f}"), state.global_step)

    def on_train_begin(self, args, state, control, **kwargs):
        """Log model architecture and hyperparameters at the beginning of training."""
        if not hasattr(args, "writer") or args.writer is None:
            return
            
        # Log hyperparameters as text
        hparams_str = "\n".join([f"{k}: {v}" for k, v in vars(args).items() 
                                if not k.startswith("_") and not callable(v)])
        args.writer.add_text("hyperparameters", hparams_str, 0)

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def prepare_hf_factual_dataset(
    tokenizer: Any,
    dataset_name: str = "data/geofakt_x_train_grpo.jsonl",
    languages: Optional[List[str]] = None,
    sample_per_language: Optional[int] = None,
    test_size: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Load a dataset from Hugging Face, format it with the factual QA template, and return train/test splits.
    
    Args:
        tokenizer: The tokenizer to use for formatting prompts
        dataset_name (str): The name of the Hugging Face dataset to load
        languages (List[str], optional): If provided, only include these languages
        sample_per_language (int, optional): If provided, sample this many examples per language
        test_size (float): Proportion of data to use for testing (default: 0.1)
        seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, test_dataset)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    logger.info(f"Loading dataset {dataset_name} from Hugging Face...")
    
    # Load the dataset
    if "json" in dataset_name:
        # If the dataset is in JSON format, load it directly
        dataset = load_dataset("json", data_files=dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    
    # Get the main split (usually 'train')
    main_split = list(dataset.keys())[0]
    data = dataset[main_split]
    
    logger.info(f"Loaded {len(data)} examples from {dataset_name}")
    
    # Filter by languages if specified
    if languages:
        logger.info(f"Filtering dataset to include only languages: {languages}")
        data = data.filter(lambda x: x["language"] in languages)
        logger.info(f"After filtering, dataset has {len(data)} examples")
    
    # Group by language
    examples_by_language = {}
    for example in data:
        lang = example["language"]
        if lang not in examples_by_language:
            examples_by_language[lang] = []
        examples_by_language[lang].append(example)
    
    # Sample per language if specified
    formatted_data = []
    for lang, examples in examples_by_language.items():
        if sample_per_language and len(examples) > sample_per_language:
            logger.info(f"Sampling {sample_per_language} examples for language {lang}")
            sampled = random.sample(examples, sample_per_language)
        else:
            sampled = examples
            logger.info(f"Using all {len(examples)} examples for language {lang}")
        
        # Add all examples
        formatted_data.extend(sampled)
    
    # Shuffle the final dataset
    random.shuffle(formatted_data)
    
    # Create a HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)
    
    logger.info(f"Created dataset with {len(dataset)} examples")
    
    # Define the prompt generation function
    def generate_factual_prompt(example):
        question = example["question"]
        answer_list = example.get("answer_list")
        #answer = example["answers"]
        answer = answer_list[0]
        
        r1_prefix = [
            {"role": "system", "content": "You are a helpful assistant. When answering a factual question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Think and reason carefully in the same language as the question (for example, if the question is in Hindi, then think and reason in Hindi).\n3. Consider multiple perspectives and potential answers before settling on your final response.\n4. Evaluate the confidence in your answer based on the information available to you.\n5. Provide the final answer clearly in the same language as the question, making sure it's well-supported by your reasoning.\n6. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."
            },
            {"role": "system", "content": "You are a helpful assistant. When answering a factual question, first think and reason in the same language as the question (for example, if question is in Hindi then think and reason in Hindi). Then, provide the final answer clearly in that same language."},
            {"role": "user", "content": f"{question} Please think carefully and return your reasoning inside <think> </think> tags, and the final direct answer inside <answer> </answer> tags."},
            {"role": "assistant", "content": "Let me think step by step.\n<think>"}
        ]
        
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "target": answer,
            "question": question,
            # Preserve additional fields that might be needed
            "language": example.get("language"),
            "region": example.get("region"),
            "topic": example.get("topic"),
            "answer_type": example.get("answer_type"),
            "id": example.get("id"),
            "answer_list": example.get("answer_list")
        }
    
    # Apply the prompt template to the dataset
    logger.info("Applying prompt template to dataset...")
    formatted_dataset = dataset.map(generate_factual_prompt)
    
    # Split into train/test
    logger.info(f"Splitting dataset into train/test with test_size={test_size}...")
    train_test_split = formatted_dataset.train_test_split(test_size=test_size, seed=seed)
    
    logger.info(f"Final split: {len(train_test_split['train'])} train examples, {len(train_test_split['test'])} test examples")
    
    return train_test_split["train"], train_test_split["test"]

def prepare_factual_dataset(dataset_path, tokenizer):
    """
    Prepare the factual QA dataset for GRPO training
    """
    # Prepare prompt template for factual QA
    def generate_factual_prompt(question, answer):
        r1_prefix = [
            {"role": "system", "content": "You are a helpful assistant. When answering a factual question, first think and reason in the same language as the question (for example, if question is in Hindi then think and reason in Hindi). Then, provide the final answer clearly in that same language."},
            {"role": "user", "content": f"{question} Please think carefully and return your reasoning inside <think> </think> tags, and the final answer inside <answer> </answer> tags."},
            {"role": "assistant", "content": "Let me think step by step.\n<think>"}
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "target": answer,
            "question":question
        }
    
    
    from datasets import Dataset
    import json

    # Open and load the JSON file
    with open('./data/factual_dataset.json', 'r') as f:
        data = json.load(f)           
    
    # Create a HuggingFace Dataset from all collected data
    train_dataset = Dataset.from_list(data['data'])

    logger.info(f"Loaded {len(train_dataset)} examples from HuggingFace dataset")
                    
    
    # Apply the prompt template to the dataset
    logger.info("Applying prompt template to dataset...")
    train_dataset = train_dataset.map(
        lambda x: generate_factual_prompt(x["question"], x["answers"])
    )
    
    # Split into train/test
    logger.info("Splitting dataset into train/test...")
    train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    
    return train_test_split["train"], train_test_split["test"]


def grpo_function(
    model_args, script_args, training_args
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Script arguments {script_args}")

    ################
    # Load tokenizer
    ################
    tokenizer_path = getattr(script_args, "tokenizer_name_or_path", None) or getattr(model_args, "model_name_or_path", None)
    model_revision = getattr(model_args, "model_revision", "main")
    trust_remote_code = getattr(model_args, "trust_remote_code", True)
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    train_dataset, test_dataset = prepare_hf_factual_dataset(
            dataset_name="data/geofakt_x_train_grpo.jsonl",
            tokenizer=tokenizer,
            test_size=0.1,
            seed=42
    )

    logger.info(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    # Log a few examples from the dataset
    for i in range(min(3, len(train_dataset))):
        logger.info(f"Example {i}:")
        logger.info(f"Prompt: {train_dataset[i]['prompt'][:100]}...")
        logger.info(f"Target: {train_dataset[i]['target']}")

    #########################
    # Instantiate GRPO trainer
    #########################
    logger.info("Initializing GRPO trainer...")
    
    # Get necessary parameters from training_args
    output_dir = getattr(training_args, "output_dir", "runs/qwen-r1-factual-qa")
    logger.info(f"Using output directory: {output_dir}")
    
    # Create GRPO config from training_args
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=getattr(training_args, "learning_rate", 5e-7),
        lr_scheduler_type=getattr(training_args, "lr_scheduler_type", "cosine"),
        warmup_ratio=getattr(training_args, "warmup_ratio", 0.03),
        max_steps=getattr(training_args, "max_steps", 100),
        per_device_train_batch_size=getattr(training_args, "per_device_train_batch_size", 1),
        gradient_accumulation_steps=getattr(training_args, "gradient_accumulation_steps", 1),
        gradient_checkpointing=getattr(training_args, "gradient_checkpointing", True),
        logging_steps=getattr(training_args, "logging_steps", 10),
        save_steps=getattr(training_args, "save_steps", 25),
        max_prompt_length=getattr(training_args, "max_prompt_length", 256),
        max_completion_length=getattr(training_args, "max_completion_length", 1024),
        num_generations=getattr(training_args, "num_generations", 8),
        beta=getattr(training_args, "beta", 0.001),
        bf16=getattr(training_args, "bf16", True),
        use_vllm=getattr(training_args, "use_vllm", False),
    )
    
    # Create PEFT config
    peft_config = None
    if getattr(model_args, "use_peft", False):
        logger.info("Initializing PEFT config")
        peft_config = get_peft_config(model_args)
    
    trainer = GRPOTrainer(
        model=getattr(model_args, "model_name_or_path", "Qwen/Qwen2.5-7B-Instruct"),
        #reward_funcs=[format_reward_func_factual, correctness_reward_func_factual],
        reward_funcs=[correctness_reward_func_factual_fuzzy, format_reward_func_factual, language_matching_reward_func_advanced],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
    )

    # Set up SummaryWriter for more detailed logging
    writer = SummaryWriter(log_dir=output_dir)
    trainer.writer = writer  # Attach to trainer for use in callbacks

    # Add detailed tensorboard callback
    detailed_tb_callback = DetailedTensorboardCallback()
    trainer.add_callback(detailed_tb_callback)
    
    # Log the learning rate schedule
    try:
        # Create temporary scheduler for visualization
        from transformers.optimization import get_scheduler
        temp_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([0.0]))], lr=grpo_config.learning_rate)
        temp_scheduler = get_scheduler(
            grpo_config.lr_scheduler_type,
            optimizer=temp_optimizer,
            num_warmup_steps=int(grpo_config.warmup_ratio * grpo_config.max_steps),
            num_training_steps=grpo_config.max_steps,
        )
        
        # Calculate and log LR schedule
        lrs = []
        for i in range(grpo_config.max_steps):
            lrs.append(temp_scheduler.get_last_lr()[0])
            temp_scheduler.step()
        
        # Log to tensorboard
        for step, lr in enumerate(lrs):
            writer.add_scalar("training/learning_rate_schedule", lr, step)
        
        logger.info(f"Logged learning rate schedule to TensorBoard")
    except Exception as e:
        logger.warning(f"Failed to log learning rate schedule: {e}")

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(grpo_config)
    if last_checkpoint is not None and getattr(grpo_config, "resume_from_checkpoint", None) is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ***'
    )
    #train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    train_result = trainer.train()
    
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    # Format loss values with more decimal places for logging
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and "loss" in key:
            logger.info(f"{key}: {value:.7f}")
        else:
            logger.info(f"{key}: {value}")

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Log final rewards to tensorboard
    try:
        reward_metrics = {}
        for i, reward_func in enumerate(trainer.reward_funcs):
            func_name = reward_func.__name__
            if hasattr(trainer, "last_rewards") and i < len(trainer.last_rewards):
                last_reward = np.mean(trainer.last_rewards[i])
                reward_metrics[f"final_{func_name}"] = last_reward
                writer.add_scalar(f"rewards/final_{func_name}", last_reward, 0)
                logger.info(f"Final reward for {func_name}: {last_reward:.7f}")
    except Exception as e:
        logger.warning(f"Failed to log final rewards: {e}")

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    try:
        training_args.distributed_state.wait_for_everyone()  # wait for all processes to load
    except:
        logger.info("No distributed state found, continuing with model saving...")

    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")

    # Save everything else on main process
    try:
        if trainer.accelerator.is_main_process:
            trainer.create_model_card()
    except:
        logger.info("Creating model card failed, continuing with saving...")
        
    # push to hub if needed
    if getattr(training_args, "push_to_hub", False):
        logger.info("Pushing to hub...")
        try:
            trainer.push_to_hub()
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")

    # Close tensorboard writer
    writer.close()
    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    
    model_args, script_args, training_args = parser.parse_args_and_config()
      
    # Set environment variables for distributed training
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Detected {torch.cuda.device_count()} GPUs. Enabling distributed training.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    # Enable TF32 for faster training on Ampere GPUs (A100)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 precision for faster training on A100 GPUs")

    # Enable TF32 for faster training on Ampere GPUs (A100) if supported
    if torch.cuda.is_available():
        try:
            # Check CUDA capabilities before enabling TF32
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 precision for faster training on A100 GPUs")
            else:
                logger.info(f"TF32 not enabled - GPU architecture ({major}.{minor}) does not support it")
        except Exception as e:
            logger.warning(f"Could not check CUDA capabilities: {e}")
            logger.info("TF32 precision not enabled")
    else:
        logger.info("CUDA not available, TF32 precision not enabled")

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
