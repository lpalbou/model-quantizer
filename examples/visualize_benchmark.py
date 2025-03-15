#!/usr/bin/env python3
"""
Script to visualize benchmark results.
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_benchmark_results(file_path):
    """
    Load benchmark results from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        The loaded benchmark results.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading benchmark results: {e}")
        return None

def plot_tokens_per_second(results, output_dir):
    """
    Plot tokens per second for each model.
    
    Args:
        results: Benchmark results.
        output_dir: Directory to save the plot.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    models = list(results.keys())
    avg_tokens_per_second = [results[model]["avg_tokens_per_second"] for model in models]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, avg_tokens_per_second, color="skyblue")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f"{height:.2f}", ha="center", va="bottom")
    
    # Add labels and title
    plt.xlabel("Model")
    plt.ylabel("Tokens per Second")
    plt.title("Average Tokens per Second by Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "tokens_per_second.png")
    plt.savefig(output_path)
    logger.info(f"Saved tokens per second plot to {output_path}")
    
    # Close figure
    plt.close()

def plot_generation_time(results, output_dir):
    """
    Plot generation time for each model.
    
    Args:
        results: Benchmark results.
        output_dir: Directory to save the plot.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    models = list(results.keys())
    total_time = [results[model]["total_time"] for model in models]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, total_time, color="lightgreen")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f"{height:.2f}s", ha="center", va="bottom")
    
    # Add labels and title
    plt.xlabel("Model")
    plt.ylabel("Total Generation Time (seconds)")
    plt.title("Total Generation Time by Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "generation_time.png")
    plt.savefig(output_path)
    logger.info(f"Saved generation time plot to {output_path}")
    
    # Close figure
    plt.close()

def plot_tokens_per_second_by_prompt(results, output_dir):
    """
    Plot tokens per second for each prompt and model.
    
    Args:
        results: Benchmark results.
        output_dir: Directory to save the plot.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract data
    models = list(results.keys())
    prompts = [r["prompt"] for r in results[models[0]]["results"]]
    
    # Create a dictionary to store tokens per second for each model and prompt
    tokens_per_second = {}
    for model in models:
        tokens_per_second[model] = [r["tokens_per_second"] for r in results[model]["results"]]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.8 / len(models)
    
    # Set positions of bars on X axis
    r = np.arange(len(prompts))
    
    # Create bars
    for i, model in enumerate(models):
        plt.bar(r + i * bar_width, tokens_per_second[model], width=bar_width, label=model)
    
    # Add labels and title
    plt.xlabel("Prompt")
    plt.ylabel("Tokens per Second")
    plt.title("Tokens per Second by Prompt and Model")
    plt.xticks(r + bar_width * (len(models) - 1) / 2, [f"Prompt {i+1}" for i in range(len(prompts))], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "tokens_per_second_by_prompt.png")
    plt.savefig(output_path)
    logger.info(f"Saved tokens per second by prompt plot to {output_path}")
    
    # Close figure
    plt.close()

def generate_html_report(results, output_dir):
    """
    Generate an HTML report of the benchmark results.
    
    Args:
        results: Benchmark results.
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
        <title>Benchmark Results</title>
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
        <h1>Benchmark Results</h1>
    """
    
    # Add summary table
    html_content += """
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Total Tokens</th>
                <th>Total Time (s)</th>
                <th>Avg Tokens/s</th>
                <th>Min Tokens/s</th>
                <th>Max Tokens/s</th>
                <th>Std Tokens/s</th>
            </tr>
    """
    
    for model in models:
        model_results = results[model]
        html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{model_results["total_tokens"]}</td>
                <td>{model_results["total_time"]:.2f}</td>
                <td>{model_results["avg_tokens_per_second"]:.2f}</td>
                <td>{model_results["min_tokens_per_second"]:.2f}</td>
                <td>{model_results["max_tokens_per_second"]:.2f}</td>
                <td>{model_results["std_tokens_per_second"]:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    """
    
    # Add charts
    html_content += """
        <h2>Charts</h2>
        <div class="chart">
            <h3>Average Tokens per Second by Model</h3>
            <img src="tokens_per_second.png" alt="Tokens per Second">
        </div>
        <div class="chart">
            <h3>Total Generation Time by Model</h3>
            <img src="generation_time.png" alt="Generation Time">
        </div>
        <div class="chart">
            <h3>Tokens per Second by Prompt and Model</h3>
            <img src="tokens_per_second_by_prompt.png" alt="Tokens per Second by Prompt">
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
                <th>Prompt</th>
                <th>Input Tokens</th>
                <th>Output Tokens</th>
                <th>New Tokens</th>
                <th>Generation Time (s)</th>
                <th>Tokens/s</th>
            </tr>
        """
        
        for result in results[model]["results"]:
            prompt = result["prompt"]
            if len(prompt) > 50:
                prompt = prompt[:50] + "..."
            
            html_content += f"""
            <tr>
                <td>{prompt}</td>
                <td>{result["input_tokens"]}</td>
                <td>{result["output_tokens"]}</td>
                <td>{result["new_tokens"]}</td>
                <td>{result["generation_time"]:.2f}</td>
                <td>{result["tokens_per_second"]:.2f}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
    
    # Close HTML content
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML content to file
    output_path = os.path.join(output_dir, "benchmark_report.html")
    with open(output_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {output_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the benchmark results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_visualizations",
        help="Directory to save the visualizations"
    )
    
    args = parser.parse_args()
    
    # Load benchmark results
    results = load_benchmark_results(args.input)
    if not results:
        return 1
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    plot_tokens_per_second(results, args.output_dir)
    plot_generation_time(results, args.output_dir)
    plot_tokens_per_second_by_prompt(results, args.output_dir)
    
    # Generate HTML report
    generate_html_report(results, args.output_dir)
    
    logger.info(f"Visualizations saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 