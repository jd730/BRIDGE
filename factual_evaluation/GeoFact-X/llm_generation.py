import json
import torch
import re
import os
import argparse
from tqdm import tqdm

# Import vLLM components
from vllm import LLM, SamplingParams

# Keep PEFT imports for LoRA handling
from peft import PeftModel, PeftConfig

# Keep transformers for tokenizer and model loading when merging LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Test model on a test dataset with vLLM acceleration")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test_dataset.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save result.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use")
    
    # vLLM specific arguments
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism in vLLM")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "awq", "squeezellm", "gptq"], 
                       help="Quantization method to use with vLLM")
    parser.add_argument("--max_context_len", type=int, default=4096, help="Maximum context length for vLLM")
    parser.add_argument("--merged_model_dir", type=str, default=None, 
                       help="Directory to save the merged LoRA model (if not provided, will use a temp directory)")
    
    return parser.parse_args()

def merge_lora_to_base_model(args):
    """
    Merge LoRA weights with the base model and save to disk for vLLM
    """
    print(f"Loading base model: {args.base_model} for LoRA merging")
    
    # Set up merged model directory
    merged_dir = args.merged_model_dir
    if merged_dir is None:
        import tempfile
        merged_dir = tempfile.mkdtemp(prefix="merged_model_")
        print(f"Created temporary directory for merged model: {merged_dir}")
    else:
        os.makedirs(merged_dir, exist_ok=True)
        
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
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

def load_vllm_model(model_path, args):
    """
    Load a model with vLLM for efficient inference
    """
    print(f"Loading model with vLLM: {model_path}")
    
    gpu_ids = [int(id) for id in args.gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Initialize vLLM engine
    vllm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": min(args.tensor_parallel_size, len(gpu_ids)),
        "trust_remote_code": True,
        "max_model_len": args.max_context_len,
    }
    
    # Add quantization if specified
    if args.quantization:
        vllm_kwargs["quantization"] = args.quantization
    
    # Initialize vLLM
    llm = LLM(**vllm_kwargs)
    
    return llm

def load_test_data(test_file):
    print(f"Loading test data from {test_file}")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data

def generate_prompts(test_data):
    """
    Generate prompts for all test examples
    """
    prompts = []
    for item in test_data:
        question = item["question"]
        prompt = create_prompt_with_template(question)
        prompts.append(prompt)
    return prompts

def create_prompt_with_template(question):
    """
    Create prompt with system messages and template
    """
    system_message1 = "You are a helpful assistant. When answering a factual question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Think and reason carefully in the same language as the question (for example, if the question is in Hindi, then think and reason in Hindi).\n3. Consider multiple perspectives and potential answers before settling on your final response.\n4. Evaluate the confidence in your answer based on the information available to you.\n5. Provide the final answer clearly in the same language as the question, making sure it's well-supported by your reasoning.\n6. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."
    
    system_message2 = "You are a helpful assistant. When answering a factual question, first think and reason in the same language as the question (for example, if question is in Hindi then think and reason in Hindi). Then, provide the final answer clearly in that same language."
    
    user_message = f"{question} Please think carefully and return your reasoning inside <think> </think> tags, and the final direct answer inside <answer> </answer> tags."
    
    # Build prompt in ChatML format that vLLM and most models understand
    prompt = f"<|im_start|>system\n{system_message1}<|im_end|>\n"
    prompt += f"<|im_start|>system\n{system_message2}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\nLet me think step by step.\n<think>"
    
    return prompt

def generate_with_vllm(llm, prompts, args, test_data):
    """
    Generate responses using vLLM's batch processing
    """
    print(f"Generating responses with vLLM in batches of {args.batch_size}...")
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=args.max_new_tokens,
        stop=["<|im_start|>", "<|im_end|>"]  # Stop on chat markers
    )
    
    # Process in batches
    all_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1}/{(len(prompts) + args.batch_size - 1)//args.batch_size}")
        
        # Generate completions for the batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process outputs
        for output in outputs:
            generated_text = output.outputs[0].text
            all_outputs.append(generated_text)
    
    # Process the outputs and return results
    results = []
    model_name = f"{args.base_model}"
    if args.lora_weights:
        model_name += f"_lora_{os.path.basename(args.lora_weights)}"
    
    for i, generation in enumerate(all_outputs):
        item = test_data[i].copy()
        
        # Process the generation to ensure it has proper structure
        llm_generation = process_generation(generation)
        
        # Extract thinking and answer parts
        think_content = ""
        answer_content = ""
        
        # Try to extract <think> content
        think_match = re.search(r'<think>(.*?)</think>', llm_generation, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        
        # Try to extract <answer> content
        answer_match = re.search(r'<answer>(.*?)</answer>', llm_generation, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        
        # Create result entry
        item["llm_generation"] = llm_generation
        item["model_name"] = model_name
        item["extracted_reasoning"] = think_content
        item["extracted_answer"] = answer_content
        
        results.append(item)
    
    return results

def process_generation(text):
    """
    Process and fix generations that might have incomplete tags
    """
    # Ensure the text has proper think and answer tags
    if "</think>" not in text:
        text += "</think>"
    
    if "<answer>" not in text:
        text += "\n<answer>Unable to generate a complete answer</answer>"
    elif "</answer>" not in text:
        text += "</answer>"
    
    return text

def save_results(results, output_file):
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    args = parse_args()
    
    # Determine whether to use vLLM (default now) or fallback to old method
    if not args.use_vllm:
        print("Warning: Not using vLLM will be significantly slower. Consider using --use_vllm for faster inference.")
        # Here you would call the original, non-vLLM code
        # For brevity, I'm not including this path as we're focusing on vLLM integration
        raise NotImplementedError("Non-vLLM path is not implemented in this version")
    
    # Determine model path (either merged with LoRA or direct base model)
    model_path = args.base_model
    
    # If LoRA weights are provided, merge them with the base model
    if args.lora_weights:
        print("LoRA weights provided, merging with base model...")
        model_path = merge_lora_to_base_model(args)
    
    # Load model with vLLM
    llm = load_vllm_model(model_path, args)
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Generate prompts for all test examples
    prompts = generate_prompts(test_data)
    
    # Generate responses using vLLM
    results = generate_with_vllm(llm, prompts, args, test_data)
    
    # Save results
    save_results(results, args.output_file)
    
    # Cleanup temporary directory if created
    if args.merged_model_dir is None and args.lora_weights is not None:
        import shutil
        try:
            shutil.rmtree(model_path)
            print(f"Removed temporary directory: {model_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {model_path}: {str(e)}")
    
    print("Done!")

if __name__ == "__main__":
    main()