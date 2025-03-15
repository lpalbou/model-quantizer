#!/usr/bin/env python3
"""
Example script demonstrating how to use the Model Quantizer.
This script shows how to quantize any Hugging Face model using different methods.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantizer.quantization_config import QuantizationConfig
from quantizer.model_quantizer import ModelQuantizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="microsoft/Phi-4-mini-instruct",
        help="Hugging Face model to quantize (default: microsoft/Phi-4-mini-instruct)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "bnb", "awq"],
        default="gptq",
        help="Quantization method to use (default: gptq)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[2, 3, 4, 8],
        default=4,
        help="Number of bits for quantization (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./quantized-model",
        help="Directory to save the quantized model (default: ./quantized-model)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for quantization (auto, cpu, cuda, mps) (default: auto)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for GPTQ quantization (default: 128)"
    )
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Comma-separated list of sentences for calibration"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the quantized model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for publishing to Hugging Face Hub"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse calibration dataset if provided
    calibration_dataset = None
    if args.calibration_dataset:
        calibration_dataset = args.calibration_dataset.split(",")
    
    # Create quantization configuration
    config = QuantizationConfig(
        bits=args.bits,
        method=args.method,
        output_dir=args.output_dir,
        group_size=args.group_size,
        device=args.device,
        calibration_dataset=calibration_dataset
    )
    
    # Create quantizer
    quantizer = ModelQuantizer(config)
    
    # Quantize model
    logger.info(f"Quantizing model {args.model} using {args.method} with {args.bits} bits")
    model, tokenizer = quantizer.quantize(args.model)
    
    # Save model
    logger.info(f"Saving quantized model to {args.output_dir}")
    quantizer.save()
    
    # Publish to Hugging Face Hub if requested
    if args.publish:
        if not args.repo_id:
            # Generate a default repo ID if not provided
            model_name = args.model.split("/")[-1]
            args.repo_id = f"{os.environ.get('HF_USERNAME', 'user')}/{model_name}-{args.method}-{args.bits}bit"
            
        logger.info(f"Publishing quantized model to {args.repo_id}")
        quantizer.publish_to_hub(repo_id=args.repo_id)
    
    logger.info("Quantization completed successfully")
    
    # Print usage instructions
    print("\nTo use the quantized model:")
    print("1. Load the model using the Transformers library:")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   model = AutoModelForCausalLM.from_pretrained(\"{args.output_dir}\", device_map=\"auto\")")
    print(f"   tokenizer = AutoTokenizer.from_pretrained(\"{args.output_dir}\")")
    print("2. Generate text:")
    print(f"   inputs = tokenizer(\"Your prompt here\", return_tensors=\"pt\").to(model.device)")
    print(f"   outputs = model.generate(**inputs, max_new_tokens=100)")
    print(f"   print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
    
    # Print benchmark instructions
    print("\nTo benchmark the quantized model:")
    print(f"python benchmark_your_model.py --original {args.model} --quantized {args.output_dir} --device {args.device}")
    
    # Print interactive testing instructions
    print("\nTo interactively test the quantized model:")
    print(f"python chat_with_model.py --model_path {args.output_dir} --device {args.device}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 