#!/usr/bin/env python3
"""
Script to compare the memory usage of different quantization methods.
"""

import os
import sys
import argparse
import logging
import time
import json
import torch
import psutil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """
    Get the current memory usage of the process.
    
    Returns:
        A dictionary with memory usage information.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_full = process.memory_full_info() if hasattr(process, 'memory_full_info') else None
    
    memory_usage = {
        "rss": memory_info.rss / (1024 ** 3),  # GB
        "vms": memory_info.vms / (1024 ** 3),  # GB
    }
    
    if memory_full:
        if hasattr(memory_full, 'uss'):
            memory_usage["uss"] = memory_full.uss / (1024 ** 3)  # GB
        if hasattr(memory_full, 'pss'):
            memory_usage["pss"] = memory_full.pss / (1024 ** 3)  # GB
    
    return memory_usage

def load_model_and_measure_memory(model_path, device="auto"):
    """
    Load a model and measure memory usage.
    
    Args:
        model_path: Path to the model.
        device: Device to load the model on.
        
    Returns:
        A dictionary with memory usage information.
    """
    # Measure initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: RSS={initial_memory['rss']:.2f} GB, VMS={initial_memory['vms']:.2f} GB")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_time = time.time() - tokenizer_start
    logger.info(f"Tokenizer loaded in {tokenizer_time:.2f} seconds")
    
    # Measure memory after loading tokenizer
    tokenizer_memory = get_memory_usage()
    logger.info(f"Memory after loading tokenizer: RSS={tokenizer_memory['rss']:.2f} GB, VMS={tokenizer_memory['vms']:.2f} GB")
    
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
    
    # Measure memory after loading model
    model_memory = get_memory_usage()
    logger.info(f"Memory after loading model: RSS={model_memory['rss']:.2f} GB, VMS={model_memory['vms']:.2f} GB")
    
    # Calculate memory used by model
    model_memory_usage = {
        "rss": model_memory["rss"] - initial_memory["rss"],
        "vms": model_memory["vms"] - initial_memory["vms"],
    }
    
    if "uss" in model_memory and "uss" in initial_memory:
        model_memory_usage["uss"] = model_memory["uss"] - initial_memory["uss"]
    
    if "pss" in model_memory and "pss" in initial_memory:
        model_memory_usage["pss"] = model_memory["pss"] - initial_memory["pss"]
    
    logger.info(f"Memory used by model: RSS={model_memory_usage['rss']:.2f} GB, VMS={model_memory_usage['vms']:.2f} GB")
    
    # Get model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e9  # billions
    logger.info(f"Model size: {model_size:.2f}B parameters")
    
    # Get model device
    model_device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {model_device}")
    
    # Run garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Measure memory after garbage collection
    gc_memory = get_memory_usage()
    logger.info(f"Memory after garbage collection: RSS={gc_memory['rss']:.2f} GB, VMS={gc_memory['vms']:.2f} GB")
    
    # Free memory
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Measure memory after freeing
    final_memory = get_memory_usage()
    logger.info(f"Final memory usage: RSS={final_memory['rss']:.2f} GB, VMS={final_memory['vms']:.2f} GB")
    
    return {
        "initial": initial_memory,
        "tokenizer": tokenizer_memory,
        "model": model_memory,
        "model_usage": model_memory_usage,
        "gc": gc_memory,
        "final": final_memory,
        "model_size": model_size,
        "device": str(model_device),
        "loading_time": model_time
    }

def plot_memory_usage(results, output_dir):
    """
    Plot memory usage for each model.
    
    Args:
        results: Memory usage results.
        output_dir: Directory to save the plot.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    models = list(results.keys())
    rss_usage = [results[model]["model_usage"]["rss"] for model in models]
    vms_usage = [results[model]["model_usage"]["vms"] for model in models]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set positions of bars on X axis
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, rss_usage, width=bar_width, label="RSS (Resident Set Size)", color="skyblue")
    plt.bar(r2, vms_usage, width=bar_width, label="VMS (Virtual Memory Size)", color="lightgreen")
    
    # Add values on top of bars
    for i, v in enumerate(rss_usage):
        plt.text(r1[i], v + 0.1, f"{v:.2f} GB", ha="center", va="bottom")
    
    for i, v in enumerate(vms_usage):
        plt.text(r2[i], v + 0.1, f"{v:.2f} GB", ha="center", va="bottom")
    
    # Add labels and title
    plt.xlabel("Model")
    plt.ylabel("Memory Usage (GB)")
    plt.title("Memory Usage by Model")
    plt.xticks([r + bar_width/2 for r in range(len(models))], models, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "memory_usage.png")
    plt.savefig(output_path)
    logger.info(f"Saved memory usage plot to {output_path}")
    
    # Close figure
    plt.close()

def generate_html_report(results, output_dir):
    """
    Generate an HTML report of the memory usage results.
    
    Args:
        results: Memory usage results.
        output_dir: Directory to save the report.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    models = list(results.keys())
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Memory Usage Comparison</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #333;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .chart {
                margin: 20px 0;
                text-align: center;
            }
            .chart img {
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <h1>Memory Usage Comparison</h1>
    """
    
    # Add summary table
    html_content += """
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>RSS (GB)</th>
                <th>VMS (GB)</th>
                <th>Model Size (B params)</th>
                <th>Device</th>
                <th>Loading Time (s)</th>
            </tr>
    """
    
    for model in models:
        model_results = results[model]
        html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{model_results["model_usage"]["rss"]:.2f}</td>
                <td>{model_results["model_usage"]["vms"]:.2f}</td>
                <td>{model_results["model_size"]:.2f}</td>
                <td>{model_results["device"]}</td>
                <td>{model_results["loading_time"]:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    """
    
    # Add chart
    html_content += """
        <h2>Chart</h2>
        <div class="chart">
            <h3>Memory Usage by Model</h3>
            <img src="memory_usage.png" alt="Memory Usage">
        </div>
    """
    
    # Add detailed results
    html_content += """
        <h2>Detailed Results</h2>
    """
    
    for model in models:
        html_content += f"""
        <h3>{model}</h3>
        <table>
            <tr>
                <th>Stage</th>
                <th>RSS (GB)</th>
                <th>VMS (GB)</th>
            </tr>
            <tr>
                <td>Initial</td>
                <td>{results[model]["initial"]["rss"]:.2f}</td>
                <td>{results[model]["initial"]["vms"]:.2f}</td>
            </tr>
            <tr>
                <td>After Loading Tokenizer</td>
                <td>{results[model]["tokenizer"]["rss"]:.2f}</td>
                <td>{results[model]["tokenizer"]["vms"]:.2f}</td>
            </tr>
            <tr>
                <td>After Loading Model</td>
                <td>{results[model]["model"]["rss"]:.2f}</td>
                <td>{results[model]["model"]["vms"]:.2f}</td>
            </tr>
            <tr>
                <td>After Garbage Collection</td>
                <td>{results[model]["gc"]["rss"]:.2f}</td>
                <td>{results[model]["gc"]["vms"]:.2f}</td>
            </tr>
            <tr>
                <td>Final</td>
                <td>{results[model]["final"]["rss"]:.2f}</td>
                <td>{results[model]["final"]["vms"]:.2f}</td>
            </tr>
        </table>
        """
    
    # Close HTML content
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML content to file
    output_path = os.path.join(output_dir, "memory_usage_report.html")
    with open(output_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {output_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare memory usage of different quantization methods")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the models to compare"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="memory_usage_comparison",
        help="Directory to save the results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for loading models (auto, cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Compare memory usage
    results = {}
    
    for model_path in args.model_paths:
        model_name = Path(model_path).name
        logger.info(f"Measuring memory usage for model: {model_name}")
        
        try:
            # Load model and measure memory
            memory_usage = load_model_and_measure_memory(model_path, args.device)
            
            # Add to results
            results[model_name] = memory_usage
            
        except Exception as e:
            logger.error(f"Error measuring memory usage for model {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Save results
    try:
        with open(os.path.join(args.output_dir, "memory_usage_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved memory usage results to {os.path.join(args.output_dir, 'memory_usage_results.json')}")
    except Exception as e:
        logger.error(f"Error saving memory usage results: {e}")
    
    # Generate visualizations
    plot_memory_usage(results, args.output_dir)
    
    # Generate HTML report
    generate_html_report(results, args.output_dir)
    
    logger.info(f"Memory usage comparison completed. Results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 