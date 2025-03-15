#!/usr/bin/env python3
"""
Script to quantize the Microsoft Phi-4-mini-instruct model.
"""

import os
import sys
import argparse
import logging
import time
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
    parser = argparse.ArgumentParser(description="Quantize the Microsoft Phi-4-mini-instruct model")
    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "bnb"],
        default="gptq",
        help="Quantization method to use (gptq or bnb)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Bit width for quantization (4 or 8)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized-models/phi4-mini",
        help="Directory to save the quantized model"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        help="Dataset to use for calibration (for GPTQ)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of samples to use for calibration (for GPTQ)"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for GPTQ quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for quantization (auto, cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create quantization config
    config = QuantizationConfig(
        model_name="microsoft/Phi-4-mini-instruct",
        method=args.method,
        bits=args.bits,
        output_dir=str(output_dir),
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        num_samples=args.num_samples,
        group_size=args.group_size,
        device=args.device
    )
    
    # Log configuration
    logger.info(f"Quantizing model with the following configuration:")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - Method: {config.method}")
    logger.info(f"  - Bits: {config.bits}")
    logger.info(f"  - Output directory: {config.output_dir}")
    logger.info(f"  - Device: {config.device}")
    if config.method == "gptq":
        logger.info(f"  - Dataset: {config.dataset}")
        logger.info(f"  - Number of samples: {config.num_samples}")
        logger.info(f"  - Group size: {config.group_size}")
    
    # Create quantizer
    quantizer = ModelQuantizer(config)
    
    # Quantize model
    start_time = time.time()
    logger.info("Starting quantization...")
    
    try:
        quantizer.quantize()
        end_time = time.time()
        logger.info(f"Quantization completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Quantized model saved to {config.output_dir}")
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 