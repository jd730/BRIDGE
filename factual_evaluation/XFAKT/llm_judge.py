import random
import csv
import os
import torch
import json
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
import pandas as pd
from utils import SYSTEM_PROMPT_OURS, PROMPT_OURS, PROMPT_ROBUST

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]

random.seed(0)

def parse_args():
    parser = ArgumentParser(description="Evaluate model outputs with vLLM acceleration")
    parser.add_argument("--model", type=str, help="Model path or name for output directories")
    parser.add_argument("--evaluator_model", type=str, default="Qwen/Qwen2.5-72B-Instruct", 
                        help="Model to use for evaluation")
    parser.add_argument("--dataset", type=str, default="factual_recall", 
                        help="Dataset type: incontext_recall/factual_recall/counter_factual")
    parser.add_argument("--prompt", type=str, default="without_system_prompt", 
                        help="Prompt style used in generation")
    parser.add_argument("--lora_weights", type=str, default=None, 
                        help="Path to LoRA weights for evaluator model (optional)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, 
                        help="Number of GPUs for tensor parallelism in vLLM")
    parser.add_argument("--swap_space", type=int, default=4, 
                        help="Swap space in GB for vLLM")
    parser.add_argument("--quantization", type=str, default=None, 
                        choices=[None, "awq", "squeezellm", "gptq"], 
                        help="Quantization method to use with vLLM")
    parser.add_argument("--max_tokens", type=int, default=256, 
                        help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for inference")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", 
                        help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--max_context_len", type=int, default=4096, 
                        help="Maximum context length for vLLM")
    
    args = parser.parse_args()
    
    # Use PROMPT_ROBUST for incontext_factual dataset
    if args.dataset == "incontext_factual":
        global PROMPT_OURS
        PROMPT_OURS = PROMPT_ROBUST
        
    # Set default prompt if not provided
    if not hasattr(args, 'prompt') or args.prompt is None:
        args.prompt = "without_system_prompt"
        
    return args

def merge_lora_to_base_model(base_model, lora_weights):
    """
    Merge LoRA weights with the base model and save to disk for vLLM
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT and transformers are required to use LoRA weights. Please install them with 'pip install peft transformers'")
    
    print(f"Loading base model: {base_model} for LoRA merging")
    
    # Set up merged model directory
    import tempfile
    merged_dir = tempfile.mkdtemp(prefix="merged_model_")
    print(f"Created temporary directory for merged model: {merged_dir}")
        
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load and merge LoRA weights
    print(f"Loading and merging LoRA weights from: {lora_weights}")
    try:
        # Load PEFT model
        model = PeftModel.from_pretrained(model, lora_weights)
        
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
    Initialize vLLM model with advanced features
    """
    # Determine model path (either merged with LoRA or direct base model)
    model_path = args.evaluator_model
    temp_dir = None
    
    # If LoRA weights are provided, merge them with the base model
    if args.lora_weights:
        print("LoRA weights provided, merging with base model...")
        temp_dir = merge_lora_to_base_model(args.evaluator_model, args.lora_weights)
        model_path = temp_dir
    
    print(f"Loading model with vLLM: {model_path}")
    
    # Configure GPU visibility
    gpu_ids = [int(id) for id in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Initialize vLLM engine with advanced parameters
    vllm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": min(args.tensor_parallel_size, len(gpu_ids)),
        "swap_space": args.swap_space,
        "trust_remote_code": True,
    }
    
    # Add max model length if specified
    if args.max_context_len:
        vllm_kwargs["max_model_len"] = args.max_context_len
    
    # Add quantization if specified
    if args.quantization:
        vllm_kwargs["quantization"] = args.quantization
    
    # Initialize vLLM
    try:
        llm = LLM(**vllm_kwargs)
        llm_tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        sampling_params.stop = [llm_tokenizer.eos_token, "<|eot_id|>"]
        
        return llm, sampling_params, temp_dir
    except Exception as e:
        print(f"Error initializing vLLM: {str(e)}")
        # Clean up temp directory if created
        if temp_dir:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        raise

def get_llm_outputs(llm, prompts, sampling_params, system_prompt, batch_size=8):
    """
    Generate outputs from LLM with batching support
    """
    # Prepare prompts with system message
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
    
    # Post-process outputs (remove prompt from the beginning)
    llm_outputs = all_outputs
    llm_outputs = [output.replace(prompt, "").strip() for prompt, output in zip(prompts, llm_outputs)]
    
    return llm_outputs

def analyze_outputs(llm_outputs, language):
    """
    Analyze LLM outputs to determine accuracy classification
    """
    results = {1: 0, 2: 0, 3: 0, 4: 0}
    for output in llm_outputs:
        if "[1]" in output:
            results[1] += 1
        elif "[2]" in output:
            if language == "English":
                results[1] += 1  # Count as Same-Correct for English
            else:
                results[2] += 1  # English-Correct for other languages
        elif "[3]" in output:
            results[3] += 1
        else:
            results[4] += 1
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize vLLM with advanced settings
    llm, sampling_params, temp_dir = init_llm(args)
    
    # Set up model name and directories
    if args.model:
        if os.path.exists(args.model):
            args.model = os.path.split(args.model)[1]
    
    # Initialize data scores dictionary
    data_scores = {args.prompt: []}
    
    # Process each language
    for lang in LANGUAGES:
        print(f"Processing language: {lang}")
        
        # Load the CSV data
        data_path = f"XFakT/generations/{args.dataset}/{args.model}/{args.prompt}/{lang}.csv"
        if not os.path.exists(data_path):
            print(f"Warning: File not found - {data_path}. Skipping language: {lang}")
            data_scores[args.prompt].append({1: 0, 2: 0, 3: 0, 4: 0})  # Add empty results
            continue
            
        data = pd.read_csv(data_path)
        
        # Prepare prompts for evaluation
        prompts = []
        for i in range(len(data)):
            prompts.append(PROMPT_OURS.format(
                question=data.iloc[i]["Question"], 
                predicted=data.iloc[i]["llm_output"], 
                answers=data.iloc[i]["answers"]
            ))
        
        # Get LLM outputs with batched processing
        llm_outputs = get_llm_outputs(llm, prompts, sampling_params, SYSTEM_PROMPT_OURS, args.batch_size)
        
        # Analyze outputs
        results = analyze_outputs(llm_outputs, lang)
        data_scores[args.prompt].append(results)
        
        # Save detailed results
        output_dir = f"XFakT/{args.dataset}/{args.model}/{args.prompt}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/{lang}.csv", "w", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Predicted", "answers", "label", "llm_output"])
            
            for i in range(len(data)):
                if "label" in data.columns:
                    label = data.iloc[i]["label"]
                else:
                    label = ""
                    
                writer.writerow([
                    data.iloc[i]["Question"], 
                    data.iloc[i]["llm_output"], 
                    data.iloc[i]["answers"], 
                    label, 
                    llm_outputs[i]
                ])
        
        print(f"Saved detailed results for {lang}")
    
    # Write summary results
    summary_path = f"XFakT/{args.dataset}/{args.model}/results_exact_match_{args.prompt}.csv"
    types = ["Same-Correct", "English-Correct", "Different-Correct", "Incorrect"]
    
    for i in range(1, 5):
        with open(summary_path, "a") as f:
            writer = csv.writer(f)
            
            if i == 1:
                writer.writerow(["Type"] + LANGUAGES)
                
            scores = []
            for j in range(len(LANGUAGES)):
                scores.append(data_scores[args.prompt][j][i])
                
            writer.writerow([types[i-1]] + scores)
    
    print(f"Saved summary results to {summary_path}")
    
    # Clean up temporary directory if LoRA was used
    if temp_dir:
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {temp_dir}: {str(e)}")
    
    print("Done!")

if __name__ == "__main__":
    main()
