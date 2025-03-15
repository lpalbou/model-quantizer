#!/bin/bash
# Script to run the benchmark and visualization process

# Function to display usage information
display_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --original MODEL_PATH    Path to the original model (required)"
    echo "  --quantized MODEL_PATH   Path to the quantized model (required)"
    echo "  --device DEVICE          Device to use for benchmarking (cpu, cuda, mps) (required)"
    echo "  --max_tokens NUM         Maximum number of tokens to generate (required)"
    echo "  --output_dir DIR         Directory to save benchmark results (required)"
    echo "  --quiet                  Run in quiet mode with minimal output"
    echo
    echo "Example:"
    echo "  $0 --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --device cpu --max_tokens 50 --output_dir benchmark_results"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    display_usage
fi

# Initialize variables
ORIGINAL_MODEL=""
QUANTIZED_MODEL=""
DEVICE=""
MAX_TOKENS=""
OUTPUT_DIR=""
QUIET=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --original)
      ORIGINAL_MODEL="$2"
      shift 2
      ;;
    --quantized)
      QUANTIZED_MODEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --quiet)
      QUIET=true
      shift
      ;;
    --help)
      display_usage
      ;;
    *)
      echo "Unknown option: $1"
      display_usage
      ;;
  esac
done

# Check if required parameters are provided
if [ -z "$ORIGINAL_MODEL" ]; then
    echo "Error: Original model path is required"
    display_usage
fi

if [ -z "$QUANTIZED_MODEL" ]; then
    echo "Error: Quantized model path is required"
    display_usage
fi

if [ -z "$DEVICE" ]; then
    echo "Error: Device is required"
    display_usage
fi

if [ -z "$MAX_TOKENS" ]; then
    echo "Error: Maximum tokens is required"
    display_usage
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory is required"
    display_usage
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$OUTPUT_DIR/benchmark_results_$TIMESTAMP.json"
REPORT_DIR="$OUTPUT_DIR/report_$TIMESTAMP"

echo "Starting benchmark process..."
echo "Original model: $ORIGINAL_MODEL"
echo "Quantized model: $QUANTIZED_MODEL"
echo "Device: $DEVICE"
echo "Max tokens: $MAX_TOKENS"
echo "Results file: $RESULTS_FILE"
echo "Report directory: $REPORT_DIR"

# Run benchmark
if [ "$QUIET" = true ]; then
  echo "Running benchmark in quiet mode..."
  python benchmark_your_model.py --original "$ORIGINAL_MODEL" --quantized "$QUANTIZED_MODEL" --device "$DEVICE" --max_new_tokens "$MAX_TOKENS" --output "$RESULTS_FILE" --quiet
else
  echo "Running benchmark..."
  python benchmark_your_model.py --original "$ORIGINAL_MODEL" --quantized "$QUANTIZED_MODEL" --device "$DEVICE" --max_new_tokens "$MAX_TOKENS" --output "$RESULTS_FILE"
fi

# Check if benchmark was successful
if [ $? -ne 0 ]; then
  echo "Benchmark failed. Exiting."
  exit 1
fi

echo "Benchmark completed. Results saved to $RESULTS_FILE"

# Generate visualization
echo "Generating visualization report..."
python visualize_benchmark.py --input "$RESULTS_FILE" --output_dir "$REPORT_DIR"

# Check if visualization was successful
if [ $? -ne 0 ]; then
  echo "Visualization failed. Exiting."
  exit 1
fi

echo "Visualization completed. Report saved to $REPORT_DIR/benchmark_report.html"

# Open the report if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Opening report..."
  open "$REPORT_DIR/benchmark_report.html"
else
  echo "Report is available at $REPORT_DIR/benchmark_report.html"
fi

echo "Benchmark process completed successfully." 