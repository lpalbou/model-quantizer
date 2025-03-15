#!/usr/bin/env python3
"""
Script to demonstrate loading and using a quantized Phi-4-mini model.
"""

import os
import sys
import argparse
import logging
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_text(model, tokenizer, prompt, system_prompt=None, max_tokens=100, temperature=0.2):
    """
    Generate text using the model.
    
    Args:
        model: The model to use for generation.
        tokenizer: The tokenizer to use for generation.
        prompt: The prompt to generate text from.
        system_prompt: The system prompt to use.
        max_tokens: The maximum number of tokens to generate.
        temperature: The temperature to use for generation.
        
    Returns:
        The generated text.
    """
    # Prepare the prompt
    if system_prompt:
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
    else:
        full_prompt = prompt
    
    # Tokenize the prompt
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
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
    
    logger.info(f"Generated {len(outputs[0]) - len(inputs['input_ids'][0])} tokens in {generation_time:.2f} seconds")
    logger.info(f"Generation speed: {(len(outputs[0]) - len(inputs['input_ids'][0])) / generation_time:.2f} tokens/second")
    
    return generated_text

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Use a quantized Phi-4-mini model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Prompt to generate text from"
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
    
    args = parser.parse_args()
    
    # Check if the model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    # Log memory usage before loading
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024 ** 3)
        logger.info(f"Initial process memory usage: {memory_usage_gb:.2f} GB")
    except ImportError:
        logger.warning("psutil not installed, skipping memory usage logging")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}...")
    start_time = time.time()
    
    # Set device map
    device_map = args.device
    if device_map == "auto":
        device_map = "auto"
    
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
        device_map=device_map,
        trust_remote_code=True,
    )
    model_time = time.time() - model_start
    logger.info(f"Model loaded in {model_time:.2f} seconds")
    
    # Log total loading time
    total_time = time.time() - start_time
    logger.info(f"Total loading time: {total_time:.2f} seconds")
    
    # Log memory usage after loading
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_gb = memory_info.rss / (1024 ** 3)
        logger.info(f"Process memory usage after loading: {memory_usage_gb:.2f} GB")
        
        # Get detailed memory info
        memory_full = process.memory_full_info()
        logger.info(f"Memory details:")
        logger.info(f"  - RSS (Resident Set Size): {memory_full.rss / (1024**3):.2f} GB")
        logger.info(f"  - VMS (Virtual Memory Size): {memory_full.vms / (1024**3):.2f} GB")
        if hasattr(memory_full, 'uss'):
            logger.info(f"  - USS (Unique Set Size): {memory_full.uss / (1024**3):.2f} GB")
        if hasattr(memory_full, 'pss'):
            logger.info(f"  - PSS (Proportional Set Size): {memory_full.pss / (1024**3):.2f} GB")
    except ImportError:
        logger.warning("psutil not installed, skipping memory usage logging")
    
    # Log model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e9  # billions
    logger.info(f"Model size: {model_size:.2f}B parameters")
    
    # Get model device
    model_device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {model_device}")
    
    # Generate text
    logger.info(f"Generating text with prompt: {args.prompt}")
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.system_prompt,
        args.max_tokens,
        args.temperature
    )
    
    # Print the generated text
    print("\nGenerated text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 