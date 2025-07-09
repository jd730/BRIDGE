import random
import csv
import os
import json
import torch
import re
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import cities, sports_persons, landmarks, politicians, festivals, artists, QUESTIONS_1, QUESTIONS_2, QUESTIONS_3, COUNTRIES

# Import PEFT for LoRA handling if needed
try:
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

random.seed(0)

LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Test multilingual factual recall with vLLM acceleration")
    parser.add_argument("--model", type=str, default="model_path", help="Path to model")
    parser.add_argument("--model_name", type=str, default="Qwen2.5 7B", help="Model name for output files")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights (optional)")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism in vLLM")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "awq", "squeezellm", "gptq"], 
                        help="Quantization method to use with vLLM")
    parser.add_argument("--max_context_len", type=int, default=4096, help="Maximum context length for vLLM")
    parser.add_argument("--dataset", type=str, default="factual_recall", help="Dataset name for output organization")
    parser.add_argument("--prompt_style", type=str, default="without_system_prompt", help="Prompt style to use")
    parser.add_argument("--output_dir", type=str, default="XFakT/generations", help="Directory to save results")
    
    return parser.parse_args()

def merge_lora_to_base_model(args):
    """
    Merge LoRA weights with the base model and save to disk for vLLM
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT and transformers are required to use LoRA weights. Please install them with 'pip install peft transformers'")
    
    print(f"Loading base model: {args.model} for LoRA merging")
    
    # Set up merged model directory
    import tempfile
    merged_dir = tempfile.mkdtemp(prefix="merged_model_")
    print(f"Created temporary directory for merged model: {merged_dir}")
        
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load and merge LoRA weights
    print(f"Loading and merging LoRA weights from: {args.lora_weights}")
    try:
        # Load PEFT model
        model = PeftModel.from_pretrained(model, args.lora_weights)
        
        # Merge weights
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
        
        # Save merged model
        print(f"Saving merged model to: {merged_dir}")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        
        return merged_dir
        
    except Exception as e:
        print(f"Error merging LoRA weights: {str(e)}")
        raise

def init_llm(args):
    """
    Initialize vLLM model with advanced features from the second code
    """
    # Determine model path (either merged with LoRA or direct base model)
    model_path = args.model
    
    # If LoRA weights are provided, merge them with the base model
    if args.lora_weights:
        print("LoRA weights provided, merging with base model...")
        model_path = merge_lora_to_base_model(args)
    
    print(f"Loading model with vLLM: {model_path}")
    
    # Configure GPU visibility
    gpu_ids = [int(id) for id in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Initialize vLLM engine with advanced parameters
    vllm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": min(args.tensor_parallel_size, len(gpu_ids)),
        "trust_remote_code": True,
    }
    
    # Add max model length if specified
    if args.max_context_len:
        vllm_kwargs["max_model_len"] = args.max_context_len
    
    # Add quantization if specified
    if args.quantization:
        vllm_kwargs["quantization"] = args.quantization
        
    # Add swap space for stability (from first code)
    vllm_kwargs["swap_space"] = 4
    
    # Initialize vLLM
    llm = LLM(**vllm_kwargs)
    
    # Get tokenizer and setup sampling parameters
    llm_tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    sampling_params.stop = [llm_tokenizer.eos_token]
    
    return llm, sampling_params, model_path

def get_llm_outputs(llm, prompts, sampling_params, system_prompt, batch_size=8):
    """
    Generate outputs from LLM with batching support
    """
    # Prepare prompts based on system prompt


    if True:
        prompts1 = []
        for prompt in prompts:
            r1_prefix = [
                    {"role": "system", "content": "You are a helpful assistant. When answering a factual question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Think and reason carefully in the same language as the question (for example, if the question is in Hindi, then think and reason in Hindi).\n3. Consider multiple perspectives and potential answers before settling on your final response.\n4. Evaluate the confidence in your answer based on the information available to you.\n5. Provide the final answer clearly in the same language as the question, making sure it's well-supported by your reasoning.\n6. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."
                    },
                    {"role": "system", "content": "You are a helpful assistant. When answering a factual question, first think and reason in the same language as the question (for example, if question is in Hindi then think and reason in Hindi). Then, provide the final answer clearly in that same language."},
                    {"role": "user", "content": f"{prompt} Please think carefully and return your reasoning inside <think> </think> tags, and the final direct answer inside <answer> </answer> tags."},
                    {"role": "assistant", "content": "Let me think step by step.\n<think>"}
                ]
            prompts1.append(r1_prefix)
    elif system_prompt == "":
        prompts1 = [[{"role": "user", "content": prompt}] for prompt in prompts]
    else:
        prompts1 = [[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}] 
                    for prompt in prompts]


    
    # Apply chat template
    tokenizer = llm.get_tokenizer()
    prompts2 = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts1]
    
    # Process in batches
    all_outputs = []
    for i in range(0, len(prompts2), batch_size):
        batch_prompts = prompts2[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts2) + batch_size - 1)//batch_size}")
        
        # Generate completions for the batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process outputs
        for output in outputs:
            generated_text = output.outputs[0].text.strip()

            all_outputs.append(generated_text)
    
    # Post-process outputs
    llm_outputs = all_outputs
    llm_outputs = [output.replace(prompt, "").strip() for prompt, output in zip(prompts, llm_outputs)]
    
    return llm_outputs

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize vLLM with advanced settings
    llm, sampling_params, model_path = init_llm(args)
    
    # Extract model name from path if needed
    args.model = os.path.split(args.model)[1]
    
    # Setup system prompt
    sys = ""  # Default for "without_system_prompt"
    
    # Prepare data
    data_all = {}
    for task in [cities, politicians, artists, landmarks, festivals, sports_persons]:
        for lang in LANGUAGES:
            task_data = task[lang]
            for i in range(len(task_data)):
                for j in range(len(LANGUAGES)):
                    if LANGUAGES[j] not in data_all:
                        data_all[LANGUAGES[j]] = []

                    if task == cities or task == landmarks:
                        ques = QUESTIONS_1[LANGUAGES[j]].format(task_data[i][j])
                        label = ("cities" if task == cities else "lakes")
                    elif task == festivals:
                        ques = QUESTIONS_3[LANGUAGES[j]].format(task_data[i][j])
                        label = ("cities" if task == cities else "lakes")
                    else:
                        ques = QUESTIONS_2[LANGUAGES[j]].format(task_data[i][j])
                        label = ("politicians" if task == politicians else "artists" if task == artists else "athletes")
                    data_all[LANGUAGES[j]].append({"question": ques, "answers": COUNTRIES[LANGUAGES[j]][lang], "label": label})

    # Create output directory
    os.makedirs(f"{args.output_dir}/{args.dataset}/{args.model_name}/{args.prompt_style}/", exist_ok=True)
    
    # Process each language
    for lang in LANGUAGES:
        print(f"Processing language: {lang}")
        data = data_all[lang]

        # Prepare prompts
        prompts = []
        for i in range(len(data)):
            prompts.append(data[i]["question"])
        
        # Get LLM outputs with batched processing
        llm_outputs = get_llm_outputs(llm, prompts, sampling_params, sys, args.batch_size)

        # Save results to CSV
        output_file = f"{args.output_dir}/{args.dataset}/{args.model_name}/{args.prompt_style}/{lang}.csv"
        with open(output_file, "w", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "answers", "label", "llm_output"])
            for i in range(len(data)):
                writer.writerow([data[i]["question"], data[i]["answers"], data[i]["label"], llm_outputs[i]])
        
        print(f"Saved results for {lang} to {output_file}")
    
    # Clean up temporary directory if LoRA was used
    if args.lora_weights and model_path != args.model:
        import shutil
        try:
            shutil.rmtree(model_path)
            print(f"Removed temporary directory: {model_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {model_path}: {str(e)}")
    
    print("Done!")

if __name__ == "__main__":
    main()
