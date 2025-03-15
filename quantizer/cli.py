"""
Command-line interface for the model quantizer.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

from .model_quantizer import ModelQuantizer
from .quantization_config import QuantizationConfig


def parse_args(args: Optional[List[str]] = None):
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments. If None, use sys.argv.
        
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Quantize and save Hugging Face models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "model_name",
        type=str,
        help="Name or path of the model to quantize"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the quantized model"
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        choices=[2, 3, 4, 8],
        default=8,
        help="Number of bits for quantization"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "awq", "bitsandbytes"],
        default="gptq",
        help="Quantization method to use"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for quantization"
    )
    
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization"
    )
    
    parser.add_argument(
        "--desc-act",
        action="store_true",
        help="Use descending activation order"
    )
    
    parser.add_argument(
        "--no-sym",
        action="store_true",
        help="Use asymmetric quantization"
    )
    
    parser.add_argument(
        "--no-optimum",
        action="store_true",
        help="Don't use the optimum library for GPTQ quantization"
    )
    
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Dataset for calibration (comma-separated list of strings or dataset name)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Hugging Face Hub publishing options
    publish_group = parser.add_argument_group("Hugging Face Hub publishing options")
    
    publish_group.add_argument(
        "--publish",
        action="store_true",
        help="Publish the quantized model to Hugging Face Hub"
    )
    
    publish_group.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for publishing (e.g., 'username/model-name')"
    )
    
    publish_group.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    publish_group.add_argument(
        "--commit-message",
        type=str,
        default="Upload quantized model",
        help="Commit message for publishing"
    )
    
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None):
    """
    Main entry point for the command-line interface.
    
    Args:
        args: Command-line arguments. If None, use sys.argv.
    """
    parsed_args = parse_args(args)
    
    # Configure logging
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse calibration dataset
    calibration_dataset = None
    if parsed_args.calibration_dataset:
        if "," in parsed_args.calibration_dataset:
            calibration_dataset = [s.strip() for s in parsed_args.calibration_dataset.split(",")]
        else:
            calibration_dataset = parsed_args.calibration_dataset
    
    # Create configuration
    config = QuantizationConfig(
        bits=parsed_args.bits,
        method=parsed_args.method,
        output_dir=parsed_args.output_dir or f"{parsed_args.model_name.split('/')[-1]}-{parsed_args.method}-{parsed_args.bits}bit",
        calibration_dataset=calibration_dataset,
        group_size=parsed_args.group_size,
        desc_act=parsed_args.desc_act,
        sym=not parsed_args.no_sym,
        device=parsed_args.device,
        use_optimum=not parsed_args.no_optimum
    )
    
    # Add model_name to config for use in model card
    config.model_name = parsed_args.model_name
    
    # Create quantizer
    quantizer = ModelQuantizer(config)
    
    try:
        # Quantize model
        model, tokenizer = quantizer.quantize(parsed_args.model_name)
        
        # Print success message
        print(f"\nModel quantized successfully and saved to: {config.output_dir}")
        print(f"Method: {config.method}")
        print(f"Bits: {config.bits}")
        
        # Publish to Hugging Face Hub if requested
        if parsed_args.publish:
            if not parsed_args.repo_id:
                # Generate a default repo ID if not provided
                model_name = parsed_args.model_name.split("/")[-1]
                repo_id = f"{os.environ.get('HF_USERNAME', 'user')}/{model_name}-{parsed_args.method}-{parsed_args.bits}bit"
                print(f"\nNo repository ID provided. Using: {repo_id}")
                print("To specify a repository ID, use the --repo-id option.")
                print("To set a default username, set the HF_USERNAME environment variable.")
            else:
                repo_id = parsed_args.repo_id
            
            # Publish model
            url = quantizer.publish_to_hub(
                repo_id=repo_id,
                private=parsed_args.private,
                commit_message=parsed_args.commit_message
            )
            
            print(f"\nModel published successfully to: {url}")
        
        # Print loading example
        print("\nTo load the quantized model:")
        print("```python")
        print("from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"model = AutoModelForCausalLM.from_pretrained(\"{config.output_dir}\", device_map=\"auto\")")
        print(f"tokenizer = AutoTokenizer.from_pretrained(\"{config.output_dir}\")")
        print("```")
        
    except Exception as e:
        logging.error(f"Error quantizing model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 