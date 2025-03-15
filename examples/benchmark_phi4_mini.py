#!/usr/bin/env python3
"""
Benchmark script to compare the performance of different quantization methods for Phi-4-mini.
"""

import os
import sys
import argparse
import logging
import time
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Benchmark prompts
BENCHMARK_PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main ingredients in a chocolate cake?",
    "Describe the process of photosynthesis.",
    "What are the benefits of regular exercise?",
    "Explain how a computer processor works.",
    "What are the key events of World War II?",
    "Describe the water cycle in nature.",
    "What are the main features of Python programming language?"
]

def load_model(model_path, device="auto"):
    """
    Load a model and tokenizer.
    
    Args:
        model_path: Path to the model.
        device: Device to load the model on.
        
    Returns:
        The loaded model and tokenizer.
    """
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_time = time.time() - tokenizer_start
    logger.info(f"Tokenizer loaded in {tokenizer_time:.2f} seconds")
    
    # Load model
    logger.info("Loading model...")
    model_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
    )
    model_time = time.time() - model_start
    logger.info(f"Model loaded in {model_time:.2f} seconds")
    
    # Log memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024 ** 3)
        logger.info(f"Process memory usage: {memory_usage_gb:.2f} GB")
    except ImportError:
        logger.warning("psutil not installed, skipping memory usage logging")
    
    # Log model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e9  # billions
    logger.info(f"Model size: {model_size:.2f}B parameters")
    
    # Get model device
    model_device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {model_device}")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, system_prompt=None, max_tokens=100, temperature=0.2):
    """
    Generate text using the model and measure performance.
    
    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for generation.
        prompt: The prompt to generate text from.
        system_prompt: The system prompt to use.
        max_tokens: The maximum number of tokens to generate.
        temperature: The temperature to use for generation.
        
    Returns:
        A dictionary with performance metrics and the generated text.
    """
    # Prepare the prompt
    if system_prompt:
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
    else:
        full_prompt = prompt
    
    # Tokenize the prompt
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate text
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    
    generation_time = time.time() - start_time
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's response if using a system prompt
    if system_prompt:
        # Find the assistant's response
        assistant_prefix = "<|assistant|>"
        assistant_start = generated_text.find(assistant_prefix)
        if assistant_start != -1:
            generated_text = generated_text[assistant_start + len(assistant_prefix):].strip()
    
    # Calculate metrics
    output_length = len(outputs[0])
    new_tokens = output_length - input_length
    tokens_per_second = new_tokens / generation_time if generation_time > 0 else 0
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "input_tokens": input_length,
        "output_tokens": output_length,
        "new_tokens": new_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second
    }

def run_benchmark(model, tokenizer, prompts, system_prompt=None, max_tokens=100, temperature=0.2):
    """
    Run a benchmark on a set of prompts.
    
    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for generation.
        prompts: The prompts to generate text from.
        system_prompt: The system prompt to use.
        max_tokens: The maximum number of tokens to generate.
        temperature: The temperature to use for generation.
        
    Returns:
        A dictionary with benchmark results.
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Running benchmark {i+1}/{len(prompts)}: {prompt[:50]}...")
        result = generate_text(model, tokenizer, prompt, system_prompt, max_tokens, temperature)
        results.append(result)
    
    # Calculate aggregate metrics
    total_tokens = sum(r["new_tokens"] for r in results)
    total_time = sum(r["generation_time"] for r in results)
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Calculate statistics
    tokens_per_second = [r["tokens_per_second"] for r in results]
    
    return {
        "results": results,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "min_tokens_per_second": min(tokens_per_second),
        "max_tokens_per_second": max(tokens_per_second),
        "std_tokens_per_second": np.std(tokens_per_second)
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark different quantization methods for Phi-4-mini")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the models to benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Path to save the benchmark results"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt to use"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature to use for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for generation (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to a JSON file with prompts to use for benchmarking"
    )
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts:
        try:
            with open(args.prompts, "r") as f:
                prompts = json.load(f)
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            return 1
    else:
        prompts = BENCHMARK_PROMPTS
    
    # Run benchmarks
    benchmark_results = {}
    
    for model_path in args.model_paths:
        model_name = Path(model_path).name
        logger.info(f"Benchmarking model: {model_name}")
        
        try:
            # Load model
            model, tokenizer = load_model(model_path, args.device)
            
            # Run benchmark
            results = run_benchmark(
                model,
                tokenizer,
                prompts,
                args.system_prompt,
                args.max_tokens,
                args.temperature
            )
            
            # Add to benchmark results
            benchmark_results[model_name] = results
            
            # Log results
            logger.info(f"Benchmark results for {model_name}:")
            logger.info(f"  - Total tokens: {results['total_tokens']}")
            logger.info(f"  - Total time: {results['total_time']:.2f} seconds")
            logger.info(f"  - Average tokens per second: {results['avg_tokens_per_second']:.2f}")
            logger.info(f"  - Min tokens per second: {results['min_tokens_per_second']:.2f}")
            logger.info(f"  - Max tokens per second: {results['max_tokens_per_second']:.2f}")
            logger.info(f"  - Std tokens per second: {results['std_tokens_per_second']:.2f}")
            
            # Free memory
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Save results
    try:
        with open(args.output, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Benchmark results saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving benchmark results: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 