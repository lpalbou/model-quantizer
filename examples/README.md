# Model Quantizer Examples

This directory contains example scripts and utilities for working with quantized models.

## Contents

- `quantize_phi4_mini.py`: Script to quantize the Microsoft Phi-4-mini-instruct model
- `update_phi4_mini_server.py`: Script to update the Phi4MiniServer to use a quantized model
- `test_quantized_phi4_mini_server.sh`: Test script to verify the updated server with a quantized model
- `use_quantized_phi4_mini.py`: Script to demonstrate loading and using a quantized model
- `benchmark_phi4_mini.py`: Script to benchmark and compare different quantization methods
- `visualize_benchmark.py`: Script to visualize benchmark results and generate HTML reports
- `compare_memory_usage.py`: Script to compare the memory usage of different quantization methods
- `custom_prompts.json`: Sample prompts for benchmarking

## Usage

### Quantizing the Phi-4-mini Model

```bash
# Quantize using GPTQ with 4-bit precision
python quantize_phi4_mini.py --method gptq --bits 4 --output_dir ./quantized-models/phi4-mini-gptq-4bit

# Quantize using BitsAndBytes with 8-bit precision
python quantize_phi4_mini.py --method bnb --bits 8 --output_dir ./quantized-models/phi4-mini-bnb-8bit
```

### Updating the Phi4MiniServer

After quantizing the model, you can update the Phi4MiniServer to use the quantized model:

```bash
python update_phi4_mini_server.py /path/to/phi4_mini_server.py /path/to/quantized/model
```

For example:

```bash
python update_phi4_mini_server.py ~/projects/notes/mnemosyne/athena/system/llm_server/phi4_mini_server.py ./quantized-models/phi4-mini-gptq-4bit
```

### Testing the Updated Server

After updating and starting the server, you can test it using the provided test script:

```bash
./test_quantized_phi4_mini_server.sh
```

This script will:
1. Check if the server is running
2. Check the memory usage
3. Test text generation
4. Check memory usage after generation

### Using a Quantized Model Directly

You can also use the quantized model directly without a server:

```bash
# Use a quantized model with default settings
python use_quantized_phi4_mini.py --model_path ./quantized-models/phi4-mini-gptq-4bit

# Specify a custom prompt
python use_quantized_phi4_mini.py --model_path ./quantized-models/phi4-mini-gptq-4bit --prompt "Explain quantum computing in simple terms."

# Adjust generation parameters
python use_quantized_phi4_mini.py --model_path ./quantized-models/phi4-mini-gptq-4bit --max_tokens 200 --temperature 0.7
```

This script will:
1. Load the quantized model and tokenizer
2. Log memory usage and model details
3. Generate text based on the provided prompt
4. Display the generated text and performance metrics

### Benchmarking Different Quantization Methods

You can benchmark and compare different quantization methods using the benchmark script:

```bash
# Benchmark two different quantized models
python benchmark_phi4_mini.py --model_paths ./quantized-models/phi4-mini-gptq-4bit ./quantized-models/phi4-mini-bnb-8bit

# Specify output file for results
python benchmark_phi4_mini.py --model_paths ./quantized-models/phi4-mini-gptq-4bit ./quantized-models/phi4-mini-bnb-8bit --output benchmark_results.json

# Use custom prompts from a JSON file
python benchmark_phi4_mini.py --model_paths ./quantized-models/phi4-mini-gptq-4bit --prompts custom_prompts.json
```

This script will:
1. Load each model and run the same set of prompts
2. Measure generation time, tokens per second, and memory usage
3. Calculate aggregate statistics (min, max, average, standard deviation)
4. Save the results to a JSON file for further analysis

### Visualizing Benchmark Results

After running benchmarks, you can visualize the results using the visualization script:

```bash
# Visualize benchmark results
python visualize_benchmark.py --input benchmark_results.json

# Specify output directory for visualizations
python visualize_benchmark.py --input benchmark_results.json --output_dir ./benchmark_report
```

This script will:
1. Generate bar charts comparing performance metrics across models
2. Create a detailed HTML report with tables and charts
3. Save all visualizations to the specified output directory

The HTML report includes:
- Summary statistics for each model
- Charts comparing tokens per second and generation time
- Detailed results for each prompt and model

### Comparing Memory Usage

You can compare the memory usage of different quantization methods using the memory comparison script:

```bash
# Compare memory usage of different quantized models
python compare_memory_usage.py --model_paths ./quantized-models/phi4-mini-gptq-4bit ./quantized-models/phi4-mini-bnb-8bit

# Specify output directory for results
python compare_memory_usage.py --model_paths ./quantized-models/phi4-mini-gptq-4bit --output_dir ./memory_comparison
```

This script will:
1. Load each model and measure memory usage at different stages
2. Generate a bar chart comparing memory usage across models
3. Create a detailed HTML report with tables and charts
4. Save all results to the specified output directory

## Memory Usage Comparison

Here's a comparison of memory usage for different quantization methods:

| Quantization Method | Approximate Memory Usage |
|---------------------|--------------------------|
| None (FP16)         | ~7.6 GB                  |
| GPTQ (8-bit)        | ~3.8 GB                  |
| GPTQ (4-bit)        | ~1.9 GB                  |
| BitsAndBytes (8-bit)| ~3.8 GB                  |
| BitsAndBytes (4-bit)| ~1.9 GB                  |

## Notes

- On macOS, GPTQ quantization generally performs better than BitsAndBytes
- For CUDA devices, BitsAndBytes is recommended
- Lower bit widths (4-bit) provide better memory efficiency but may slightly reduce output quality
- The memory usage values are approximate and may vary based on your system configuration 