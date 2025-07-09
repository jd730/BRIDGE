import json
import re
import os
import argparse
from typing import List, Dict, Any
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict

# Import vLLM components
from vllm import LLM, SamplingParams

# Keep transformers for tokenizer handling
from transformers import AutoTokenizer

# Keep PEFT imports for LoRA handling
from peft import PeftModel, PeftConfig

# Keep transformers for tokenizer and model loading when merging LoRA
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM reasoning quality with vLLM acceleration")
    parser.add_argument("--results_file", type=str, required=True, help="Path to results.json or results.jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save evaluation results (.json or .jsonl)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Evaluation model name")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens for generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--output_format", choices=["json", "jsonl"], help="Force specific output format (default: based on output_file extension)")
    
    # vLLM specific arguments
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism in vLLM")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "awq", "squeezellm", "gptq"], 
                       help="Quantization method to use with vLLM")
    parser.add_argument("--max_context_len", type=int, default=4096, help="Maximum context length for vLLM")
    parser.add_argument("--merged_model_dir", type=str, default=None, 
                       help="Directory to save the merged LoRA model (if not provided, will use a temp directory)")
    
    # Analysis arguments
    parser.add_argument("--analysis_output_dir", type=str, default=None, 
                       help="Directory to save language vs region analysis results")
    
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

def load_evaluation_model_vllm(args):
    """Load the evaluation model with vLLM for batched inference."""
    # Determine model path (either merged with LoRA or direct base model)
    model_path = args.base_model
    
    # If LoRA weights are provided, merge them with the base model
    if args.lora_weights:
        print("LoRA weights provided, merging with base model...")
        model_path = merge_lora_to_base_model(args)
    
    # Load model with vLLM
    llm = load_vllm_model(model_path, args)
    
    # Load tokenizer directly - we still need it for prompt creation
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        trust_remote_code=True,
        use_fast=True,
    )
    
    return llm, tokenizer, model_path

def load_evaluation_model_hf(args):
    """Load the evaluation model and tokenizer using HuggingFace (fallback)."""
    print(f"Loading evaluation model with HuggingFace: {args.base_model}")
    
    # Set device map
    device_map = "auto" if torch.cuda.device_count() > 1 else None
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map=device_map,
    )
    
    # If not using device_map, manually move to device
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        trust_remote_code=True,
        use_fast=True,
    )
    
    return model, tokenizer, None

def load_results(results_file):
    """Load the results from the JSONL file."""
    print(f"Loading results from {results_file}")
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    
    print(f"Loaded {len(results)} results")
    return results

def extract_reasoning_and_answer(result):
    """Extract the ground truth reasoning and LLM reasoning/answer from a result entry."""
    # Ground truth reasoning from the "reasoning" field
    ground_truth_reasoning = result.get("reasoning", "")
    llm_generation = result.get("llm_generation", "")
    
    return ground_truth_reasoning, llm_generation

def create_evaluation_prompt(question: str, answer_list: List[str], ground_truth_reasoning: str, llm_generation: str):
    """Create a prompt for the evaluation model."""
    prompt = f"""# Reasoning Quality Evaluation
    You are an expert reasoning evaluator tasked with comparing an LLM's reasoning trace against a ground truth reasoning trace. Your evaluation must be fair, consistent, and based solely on the quality of reasoning, not on superficial similarities.

    ## Input:
    - Question: {question}
    - Answer List: {answer_list}
    - Ground Truth Reasoning: {ground_truth_reasoning}
    - LLM Response: {llm_generation}


    ## Evaluation Criteria:
    Assess the quality of the LLM's reasoning compared to the ground truth on a scale from 0-10 based on the following:

    1. Logical Structure (40%):
    - How well does the reasoning follow a clear, step-by-step logical progression?
    - Are the steps in a sensible order that builds toward the answer?

    2. Key Insights (30%):
    - Does the reasoning identify the same critical insights as the ground truth?
    - Are the important clues from the question properly recognized and utilized?

    3. Factual Correctness (20%):
    - Is the reasoning free from factual errors?
    - Does it avoid adding irrelevant information or missing necessary information?

    4. Conclusion Validity (10%):
    - Does the reasoning correctly lead to the answer?
    - Is the link between the reasoning and the conclusion clear?

    ## Scoring Guide:
    0-1: Completely irrelevant or fundamentally flawed reasoning
    2-3: Major logical errors or missing critical insights
    4-5: Contains some correct elements but misses important aspects
    6-7: Good reasoning with minor gaps or imperfections
    8-9: Very good reasoning, almost matching ground truth quality
    10: Perfect reasoning, capturing all key insights with proper structure

    ## Language Mismatch:
    Determine if the LLM reasoning is in a different language than the ground truth reasoning.
    - 0: Same language
    - 1: Different language

    ## Your Response (FORMAT STRICTLY REQUIRED):
    REASONING_SCORE: [integer between 0-10]  
    LANGUAGE_MISMATCH: [0 or 1]  
    ANSWER_CORRECT: [0 or 1]  # 1 if the LLM's final answer matches the ground truth answers in Answer List, 0 otherwise  
    JUSTIFICATION: [Brief explanation of your evaluation, highlighting strengths and weaknesses]
    """
    return prompt

def generate_prompts(results):
    """Generate prompts for all evaluation examples."""
    prompts = []
    for result in results:
        question = result["question"]
        answer_list = result.get("answer_list", [])
        ground_truth_reasoning, llm_generation = extract_reasoning_and_answer(result)
        
        # Create evaluation prompt
        prompt = create_evaluation_prompt(question, answer_list, ground_truth_reasoning, llm_generation)
        
        # Format as chat messages (if needed by the model)
        messages = [{"role": "user", "content": prompt}]
        prompts.append((result, messages))
    
    return prompts

def parse_evaluation_output(output: str) -> Dict[str, Any]:
    """Parse the evaluation model's output to extract scores and justification."""
    result = {
        "reasoning_score": None,
        "language_mismatch": None,
        "answer_correctness": None,
        "justification": ""
    }
    
    # Extract reasoning score
    score_match = re.search(r'REASONING_SCORE:\s*(\d+)', output, re.IGNORECASE)
    if score_match:
        result["reasoning_score"] = int(score_match.group(1))
    
    # Extract language mismatch
    lang_match = re.search(r'LANGUAGE_MISMATCH:\s*([01])', output, re.IGNORECASE)
    if lang_match:
        result["language_mismatch"] = int(lang_match.group(1))

    # Extract answer correctness
    answer_match = re.search(r'ANSWER_CORRECT:\s*([01])', output, re.IGNORECASE)
    if answer_match:
        result["answer_correctness"] = int(answer_match.group(1))

    # Extract justification
    just_match = re.search(r'JUSTIFICATION:\s*(.*?)(?:\n\n|$)', output, re.DOTALL | re.IGNORECASE)
    if just_match:
        result["justification"] = just_match.group(1).strip()
    else:
        # If we can't find the formatted justification, use the rest of the text after the scores
        parts = output.split("LANGUAGE_MISMATCH:", 1)
        if len(parts) > 1:
            after_mismatch = parts[1].strip()
            after_mismatch = re.sub(r'^\s*[01]\s*', '', after_mismatch, count=1)
            result["justification"] = after_mismatch.strip()
    
    return result

def evaluate_with_vllm(llm, tokenizer, prompt_data, args):
    """Run evaluations using vLLM's batch processing."""
    print(f"Generating evaluations with vLLM in batches of {args.batch_size}...")
    
    # Extract prompts and results from prompt_data
    results = [data[0] for data in prompt_data]
    message_prompts = [data[1] for data in prompt_data]
    
    # Convert message prompts to text prompts
    text_prompts = []
    for messages in message_prompts:
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        text_prompts.append(chat_prompt)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for more deterministic results
        max_tokens=args.max_new_tokens,
        stop=["<|im_start|>", "<|im_end|>"]  # Stop on chat markers if applicable
    )
    
    # Process in batches
    all_outputs = []
    all_evaluations = []
    
    for i in range(0, len(text_prompts), args.batch_size):
        batch_prompts = text_prompts[i:i+args.batch_size]
        batch_results = results[i:i+args.batch_size]
        
        print(f"Processing batch {i//args.batch_size + 1}/{(len(text_prompts) + args.batch_size - 1)//args.batch_size}")
        
        # Generate evaluations for the batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process outputs
        for j, output in enumerate(outputs):
            result = batch_results[j]
            generated_text = output.outputs[0].text
            
            # Parse the evaluation output
            evaluation_result = parse_evaluation_output(generated_text)
            
            # Add ground truth answers for comparison
            evaluation_result["ground_truth_answers"] = result.get("answer_list", [])
            evaluation_result["corrected_answers"] = result.get("corrected_answer_list", [])
            
            # Combine the original result with the evaluation
            full_result = result.copy()
            full_result["evaluation"] = evaluation_result
            all_evaluations.append(full_result)
    
    return all_evaluations

def evaluate_with_hf(model, tokenizer, prompt_data, args):
    """Fallback to evaluate using HuggingFace Transformers directly."""
    evaluation_results = []
    
    print(f"Running evaluations with HuggingFace Transformers...")
    for result, messages in tqdm(prompt_data):
        try:
            # Format as chat template
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = tokenizer(chat_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate evaluation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )
            
            # Decode output
            output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Parse the evaluation output
            evaluation_result = parse_evaluation_output(output_text)
            
            # Add ground truth answers for comparison
            evaluation_result["ground_truth_answers"] = result.get("answer_list", [])
            evaluation_result["corrected_answers"] = result.get("corrected_answer_list", [])
            
            # Combine the original result with the evaluation
            full_result = result.copy()
            full_result["evaluation"] = evaluation_result
            evaluation_results.append(full_result)
            
        except Exception as e:
            print(f"Error evaluating result: {e}")
            import traceback
            traceback.print_exc()
    
    return evaluation_results

def compute_statistics(evaluation_results):
    """Compute aggregate statistics from the evaluations."""
    stats = {
        "total_evaluations": len(evaluation_results),
        "average_reasoning_score": 0,
        "language_mismatch_count": 0,
        "answer_correctness_count": 0,
        "score_distribution": {i: 0 for i in range(11)}  # 0-10 scores
    }
    
    total_score = 0
    for result in evaluation_results:
        eval_data = result.get("evaluation", {})
        score = eval_data.get("reasoning_score")
        if score is not None:
            total_score += score
            stats["score_distribution"][score] += 1
        
        if eval_data.get("language_mismatch") == 1:
            stats["language_mismatch_count"] += 1
        
        if eval_data.get("answer_correctness") == 1:
            stats["answer_correctness_count"] += 1
    
    if stats["total_evaluations"] > 0:
        stats["average_reasoning_score"] = total_score / stats["total_evaluations"]
    
    return stats

def analyze_language_region_matrices(evaluation_results, output_dir=None):
    """
    Analyze evaluation results to create matrices of average scores by language and region.
    
    Args:
        evaluation_results: List of evaluation result objects
        output_dir: Directory to save the output files (optional)
    """
    print("\nAnalyzing language vs region performance...")
    
    # Data structures to collect information
    data_by_lang_region = defaultdict(lambda: defaultdict(list))
    
    # Metrics to track
    metrics = ['reasoning_score', 'language_mismatch', 'answer_correctness']
    
    # Set of all languages and regions for matrix dimensions
    all_languages = set()
    all_regions = set()
    
    # Process each evaluation result
    for result in evaluation_results:
        # Extract language, region, and evaluation scores
        language = result.get('language', 'unknown')
        region = result.get('region', 'unknown')
        evaluation = result.get('evaluation', {})
        
        # Skip if evaluation data is missing
        if not evaluation:
            continue
        
        # Add to our sets
        all_languages.add(language)
        all_regions.add(region)
        
        # Collect the scores
        data_point = {
            'reasoning_score': evaluation.get('reasoning_score', None),
            'language_mismatch': evaluation.get('language_mismatch', None),
            'answer_correctness': evaluation.get('answer_correctness', None)
        }
        
        # Skip entries with missing metrics
        if None in data_point.values():
            continue
            
        # Add to our data collection
        data_by_lang_region[language][region].append(data_point)
    
    print(f"Found {len(all_languages)} languages and {len(all_regions)} regions")
    
    # Convert to sorted lists for consistent matrix indices
    all_languages = sorted(all_languages)
    all_regions = sorted(all_regions)
    
    # Create DataFrames for each metric
    matrices = {}
    for metric in metrics:
        # Initialize a matrix of NaN values
        matrix = np.full((len(all_languages), len(all_regions)), np.nan)
        
        # Fill in the matrix with average values
        for i, lang in enumerate(all_languages):
            for j, region in enumerate(all_regions):
                scores = [item[metric] for item in data_by_lang_region[lang][region] 
                         if item[metric] is not None]
                if scores:
                    matrix[i, j] = sum(scores) / len(scores)
        
        # Create a DataFrame for better visualization
        df = pd.DataFrame(matrix, index=all_languages, columns=all_regions)
        matrices[metric] = df
        
        # Print summary
        print(f"\nAverage {metric} by Language and Region:")
        print(df)
        
        # Save to CSV if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"avg_{metric}_by_lang_region.csv")
            df.to_csv(output_file)
            print(f"Saved to {output_file}")
    
    return matrices

def save_results(evaluation_results, stats, output_file, matrices=None, output_format=None):
    """Save the evaluation results, statistics, and matrices to files."""
    output = {
        "statistics": stats,
        "results": evaluation_results
    }
    
    print(f"Saving evaluation results to {output_file}")
    
    # Determine file format based on extension or explicit parameter
    if output_format:
        file_format = output_format
    else:
        file_format = "jsonl" if output_file.endswith('.jsonl') else "json"
    
    if file_format == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write each result as a separate line
            for result in evaluation_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Write statistics to a separate file
        stats_file = output_file.replace('.jsonl', '_stats.json')
        if stats_file == output_file:  # In case the output file doesn't end with .jsonl
            stats_file = output_file + ".stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Statistics saved separately to {stats_file}")
    else:
        # Standard JSON output with both results and stats
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    
    # Set environment for GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Load results from the test run
    results = load_results(args.results_file)
    
    # Generate prompts for all results
    prompt_data = generate_prompts(results)
    
    # Determine whether to use vLLM or HuggingFace
    if args.use_vllm:
        # Load model with vLLM
        llm, tokenizer, model_path = load_evaluation_model_vllm(args)
        
        # Run evaluations with vLLM
        evaluation_results = evaluate_with_vllm(llm, tokenizer, prompt_data, args)
        
        # Cleanup temporary directory if created
        if args.merged_model_dir is None and args.lora_weights is not None and model_path != args.base_model:
            import shutil
            try:
                shutil.rmtree(model_path)
                print(f"Removed temporary directory: {model_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {model_path}: {str(e)}")
    else:
        print("Warning: Not using vLLM will be significantly slower. Consider using --use_vllm for faster inference.")
        # Load model with HuggingFace
        model, tokenizer, _ = load_evaluation_model_hf(args)
        
        # Run evaluations with HuggingFace
        evaluation_results = evaluate_with_hf(model, tokenizer, prompt_data, args)
    
    # Compute statistics
    stats = compute_statistics(evaluation_results)
    print("\nEvaluation Statistics:")
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Average reasoning score: {stats['average_reasoning_score']:.2f}")
    print(f"Language mismatch count: {stats['language_mismatch_count']}")
    print(f"Answer correctness count: {stats['answer_correctness_count']}")
    print("Score distribution:")
    for score, count in stats["score_distribution"].items():
        percentage = count/stats['total_evaluations']*100 if stats['total_evaluations'] > 0 else 0
        print(f"  Score {score}: {count} ({percentage:.1f}%)")
    
    # Analyze by language and region if output directory is provided
    matrices = None
    if args.analysis_output_dir:
        matrices = analyze_language_region_matrices(evaluation_results, args.analysis_output_dir)
    
    # Save results
    save_results(evaluation_results, stats, args.output_file, matrices, args.output_format)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()