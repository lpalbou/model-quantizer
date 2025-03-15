#!/usr/bin/env python3
"""
Script to run the benchmark and visualization process
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmark and visualization process")
    parser.add_argument("--original", required=True, help="Path to the original model (required)")
    parser.add_argument("--quantized", required=True, help="Path to the quantized model (required)")
    parser.add_argument("--device", required=True, help="Device to use for benchmarking (cpu, cuda, mps) (required)")
    parser.add_argument("--max_tokens", required=True, type=int, help="Maximum number of tokens to generate (required)")
    parser.add_argument("--output_dir", required=True, help="Directory to save benchmark results (required)")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode with minimal output")
    parser.add_argument("--update-model-card", action="store_true", help="Update the model card with benchmark results")
    
    return parser.parse_args()

def run_benchmark(args):
    """Run the benchmark process."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    report_dir = os.path.join(args.output_dir, f"report_{timestamp}")
    
    print("Starting benchmark process...")
    print(f"Original model: {args.original}")
    print(f"Quantized model: {args.quantized}")
    print(f"Device: {args.device}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Results file: {results_file}")
    print(f"Report directory: {report_dir}")
    if args.update_model_card:
        print("Model card will be updated with benchmark results")
    
    # Run benchmark
    benchmark_cmd = [
        "benchmark-model",
        "--original", args.original,
        "--quantized", args.quantized,
        "--device", args.device,
        "--max_new_tokens", str(args.max_tokens),
        "--output", results_file
    ]
    
    if args.quiet:
        benchmark_cmd.append("--quiet")
        print("Running benchmark in quiet mode...")
    else:
        print("Running benchmark...")
    
    if args.update_model_card:
        benchmark_cmd.append("--update-model-card")
    
    benchmark_result = subprocess.run(benchmark_cmd, check=False)
    
    if benchmark_result.returncode != 0:
        print("Benchmark failed. Exiting.")
        return False
    
    print(f"Benchmark completed. Results saved to {results_file}")
    
    # Generate visualization
    print("Generating visualization report...")
    visualize_cmd = [
        "visualize-benchmark",
        "--input", results_file,
        "--output_dir", report_dir
    ]
    
    visualize_result = subprocess.run(visualize_cmd, check=False)
    
    if visualize_result.returncode != 0:
        print("Visualization failed. Exiting.")
        return False
    
    report_path = os.path.join(report_dir, "benchmark_report.html")
    print(f"Visualization completed. Report saved to {report_path}")
    
    # Open the report if on macOS
    if sys.platform == "darwin":
        print("Opening report...")
        subprocess.run(["open", report_path], check=False)
    else:
        print(f"Report is available at {report_path}")
    
    print("Benchmark process completed successfully.")
    return True

def main_cli():
    """Entry point for the command-line interface."""
    args = parse_args()
    success = run_benchmark(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main_cli() 