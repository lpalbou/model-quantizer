#!/usr/bin/env python3
"""
Visualize benchmark results from benchmark_model.py.
Creates HTML reports with charts comparing original and quantized models.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_benchmark_results(file_path):
    """Load benchmark results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_memory_chart(results, output_dir):
    """Create a chart comparing memory usage."""
    orig_mem = results["original"]["stats"]["memory"]
    quant_mem = results["quantized"]["stats"]["memory"]
    
    metrics = ["initial_memory", "min_memory", "max_memory", "avg_memory", "memory_increase"]
    labels = ["Initial", "Min", "Max", "Avg", "Increase"]
    
    orig_values = [orig_mem[m] for m in metrics]
    quant_values = [quant_mem[m] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, orig_values, width, label='Original')
    rects2 = ax.bar(x + width/2, quant_values, width, label='Quantized')
    
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'))
    plt.close()

def create_performance_chart(results, output_dir):
    """Create a chart comparing performance metrics."""
    orig_perf = results["original"]["stats"]["performance"]
    quant_perf = results["quantized"]["stats"]["performance"]
    
    metrics = [
        ("prompt_eval_rate", "Prompt Processing"),
        ("eval_rate", "Token Generation")
    ]
    
    labels = [m[1] for m in metrics]
    orig_values = [orig_perf[m[0]] for m in metrics]
    quant_values = [quant_perf[m[0]] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, orig_values, width, label='Original')
    rects2 = ax.bar(x + width/2, quant_values, width, label='Quantized')
    
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()

def create_category_chart(results, output_dir):
    """Create a chart comparing performance by prompt category."""
    # Group results by category
    categories = {}
    for result in results["original"]["results"]:
        category = result["prompt_category"]
        if category not in categories:
            categories[category] = {"original": [], "quantized": []}
        categories[category]["original"].append(result)
    
    for result in results["quantized"]["results"]:
        category = result["prompt_category"]
        categories[category]["quantized"].append(result)
    
    # Calculate average tokens per second for each category
    category_labels = list(categories.keys())
    orig_values = []
    quant_values = []
    
    for category in category_labels:
        orig_tokens_per_sec = np.mean([r["tokens_per_second"] for r in categories[category]["original"]])
        quant_tokens_per_sec = np.mean([r["tokens_per_second"] for r in categories[category]["quantized"]])
        orig_values.append(orig_tokens_per_sec)
        quant_values.append(quant_tokens_per_sec)
    
    x = np.arange(len(category_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, orig_values, width, label='Original')
    rects2 = ax.bar(x + width/2, quant_values, width, label='Quantized')
    
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Performance by Prompt Category')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in category_labels])
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'))
    plt.close()

def create_html_report(results, output_dir):
    """Create an HTML report with benchmark results."""
    original_model = os.path.basename(results["config"]["original_model"])
    quantized_model = os.path.basename(results["config"]["quantized_model"])
    timestamp = datetime.fromisoformat(results["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate summary metrics
    orig_mem = results["original"]["stats"]["memory"]
    quant_mem = results["quantized"]["stats"]["memory"]
    memory_reduction = (1 - (quant_mem["max_memory"] / orig_mem["max_memory"])) * 100
    
    orig_perf = results["original"]["stats"]["performance"]
    quant_perf = results["quantized"]["stats"]["performance"]
    speed_change = ((quant_perf["eval_rate"] / orig_perf["eval_rate"]) - 1) * 100
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Benchmark Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .summary {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .chart {{
                margin: 30px 0;
                text-align: center;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .highlight {{
                font-weight: bold;
                color: #2980b9;
            }}
            .positive {{
                color: green;
            }}
            .negative {{
                color: red;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Benchmark Results</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Original Model:</strong> {original_model}</p>
                <p><strong>Quantized Model:</strong> {quantized_model}</p>
                <p><strong>Benchmark Date:</strong> {timestamp}</p>
                <p><strong>Device:</strong> {results["config"]["device"]}</p>
                <p><strong>Memory Reduction:</strong> <span class="highlight">{memory_reduction:.2f}%</span></p>
                <p><strong>Generation Speed Change:</strong> <span class="{
                    'positive' if speed_change > 0 else 'negative'
                }">{speed_change:.2f}%</span></p>
            </div>
            
            <h2>Charts</h2>
            
            <div class="chart">
                <h3>Memory Usage Comparison</h3>
                <img src="memory_comparison.png" alt="Memory Usage Comparison" style="max-width: 100%;">
            </div>
            
            <div class="chart">
                <h3>Performance Comparison</h3>
                <img src="performance_comparison.png" alt="Performance Comparison" style="max-width: 100%;">
            </div>
            
            <div class="chart">
                <h3>Performance by Prompt Category</h3>
                <img src="category_comparison.png" alt="Performance by Prompt Category" style="max-width: 100%;">
            </div>
            
            <h2>Detailed Results</h2>
            
            <h3>Memory Usage (GB)</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Original</th>
                    <th>Quantized</th>
                    <th>Difference</th>
                    <th>Ratio</th>
                </tr>
    """
    
    # Add memory metrics
    metrics = ["initial_memory", "min_memory", "max_memory", "avg_memory", "memory_increase"]
    labels = ["Initial Memory", "Min Memory", "Max Memory", "Avg Memory", "Memory Increase"]
    
    for i, metric in enumerate(metrics):
        orig_val = orig_mem[metric]
        quant_val = quant_mem[metric]
        diff = quant_val - orig_val
        ratio = quant_val / orig_val if orig_val != 0 else float('inf')
        
        html_content += f"""
                <tr>
                    <td>{labels[i]}</td>
                    <td>{orig_val:.2f}</td>
                    <td>{quant_val:.2f}</td>
                    <td>{diff:.2f}</td>
                    <td>{ratio:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Performance Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Original</th>
                    <th>Quantized</th>
                    <th>Difference</th>
                    <th>Ratio</th>
                </tr>
    """
    
    # Add performance metrics
    perf_metrics = [
        ("load_duration", "Load Time (s)"),
        ("prompt_eval_count", "Prompt Tokens"),
        ("prompt_eval_duration", "Prompt Eval Time (s)"),
        ("prompt_eval_rate", "Prompt Tokens/Sec"),
        ("eval_count", "Generated Tokens"),
        ("eval_duration", "Generation Time (s)"),
        ("eval_rate", "Generation Tokens/Sec")
    ]
    
    for key, label in perf_metrics:
        orig_val = orig_perf[key]
        quant_val = quant_perf[key]
        diff = quant_val - orig_val
        ratio = quant_val / orig_val if orig_val != 0 else float('inf')
        
        html_content += f"""
                <tr>
                    <td>{label}</td>
                    <td>{orig_val:.2f}</td>
                    <td>{quant_val:.2f}</td>
                    <td>{diff:.2f}</td>
                    <td>{ratio:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Results by Prompt Category</h3>
    """
    
    # Group results by category
    categories = {}
    for result in results["original"]["results"]:
        category = result["prompt_category"]
        if category not in categories:
            categories[category] = {"original": [], "quantized": []}
        categories[category]["original"].append(result)
    
    for result in results["quantized"]["results"]:
        category = result["prompt_category"]
        categories[category]["quantized"].append(result)
    
    # Add category tables
    for category, data in categories.items():
        html_content += f"""
            <h4>{category.capitalize()} Prompts</h4>
            <table>
                <tr>
                    <th>Prompt</th>
                    <th>Original Tokens/Sec</th>
                    <th>Quantized Tokens/Sec</th>
                    <th>Difference</th>
                    <th>Ratio</th>
                </tr>
        """
        
        for i in range(len(data["original"])):
            orig_result = data["original"][i]
            quant_result = data["quantized"][i]
            
            orig_tps = orig_result["tokens_per_second"]
            quant_tps = quant_result["tokens_per_second"]
            diff = quant_tps - orig_tps
            ratio = quant_tps / orig_tps if orig_tps != 0 else float('inf')
            
            html_content += f"""
                <tr>
                    <td>{orig_result["prompt_name"]}</td>
                    <td>{orig_tps:.2f}</td>
                    <td>{quant_tps:.2f}</td>
                    <td>{diff:.2f}</td>
                    <td>{ratio:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(output_dir, 'benchmark_report.html'), 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with benchmark results")
    parser.add_argument("--output_dir", type=str, default="benchmark_report", help="Output directory for report")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark results
    results = load_benchmark_results(args.input)
    
    # Create charts
    create_memory_chart(results, args.output_dir)
    create_performance_chart(results, args.output_dir)
    create_category_chart(results, args.output_dir)
    
    # Create HTML report
    create_html_report(results, args.output_dir)
    
    print(f"Report generated in {args.output_dir}/benchmark_report.html")

if __name__ == "__main__":
    main() 