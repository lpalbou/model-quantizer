# Benchmarking Guide

This guide explains how to benchmark quantized models to evaluate their performance, memory usage, and output quality.

## Introduction

Benchmarking is a crucial step in the model quantization process. It helps you:

1. Verify that the quantized model maintains acceptable performance
2. Understand the trade-offs between different quantization methods and bit widths
3. Make informed decisions about which quantized model to use for your specific use case
4. Gather metrics to include in your model card when publishing

## Available Benchmarking Tools

The Model Quantizer provides several tools for benchmarking:

1. `benchmark-model`: A comprehensive script for comparing original and quantized models
2. `visualize-benchmark`: A script for visualizing benchmark results
3. `run-benchmark`: A shell script that automates the benchmarking and visualization process

## Choosing Between run-benchmark and benchmark-model

The Model Quantizer provides two main tools for benchmarking: `run-benchmark` and `benchmark-model`. Understanding when to use each will help you get the most out of the benchmarking process.

### When to Use run-benchmark

`run-benchmark` is an all-in-one solution that:
- Runs the benchmark comparing original and quantized models
- Saves the results to a JSON file
- Generates a visual HTML report
- Opens the report (on macOS) or provides the path to the report

Use `run-benchmark` when:
- You want a complete end-to-end benchmarking solution
- You need visual reports generated automatically
- You're doing a one-off comparison between models
- You want a simple, streamlined process

Example use cases:
```bash
# Quick comparison between original and quantized models with visual report
run-benchmark --original microsoft/Phi-4-mini-instruct --quantized ./phi4-mini-gptq-4bit --device cpu --max_tokens 50 --output_dir benchmark_results

# Comparing models for a presentation or documentation
run-benchmark --original google/gemma-2b --quantized ./gemma-2b-quantized --device cuda --max_tokens 100 --output_dir presentation_benchmarks
```

### When to Use benchmark-model

`benchmark-model` is a more flexible, lower-level tool that:
- Runs benchmarks with more customizable parameters
- Outputs raw benchmark data
- Can be integrated into custom workflows

Use `benchmark-model` when:
- You need more granular control over benchmark parameters
- You want to run multiple benchmarks and analyze the results yourself
- You're integrating benchmarking into a custom workflow or script
- You want to compare multiple models or configurations systematically
- You need to save raw benchmark data for custom analysis

Example use cases:
```bash
# Fine-tuning benchmark parameters
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./phi4-mini-gptq-4bit --device cpu --max_new_tokens 100 --temperature 0.7 --top_p 0.9 --output custom_benchmark.json

# Comparing multiple quantization methods
benchmark-model --original ./phi4-mini-gptq-8bit --quantized ./phi4-mini-gptq-4bit --output gptq_comparison.json
benchmark-model --original ./phi4-mini-bnb-8bit --quantized ./phi4-mini-bnb-4bit --output bnb_comparison.json

# Batch testing with different prompt sets
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./phi4-mini-gptq-4bit --prompts-file scientific_prompts.json --output scientific_benchmark.json
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./phi4-mini-gptq-4bit --prompts-file creative_prompts.json --output creative_benchmark.json

# Integration into a custom script
for model in ./models/*; do
    benchmark-model --original microsoft/Phi-4-mini-instruct --quantized $model --output results/$(basename $model).json
done
```

## Basic Usage

### Using the Automated Benchmark Script (Recommended)

The `run-benchmark` script provides a convenient one-step process for benchmarking and visualization:

```bash
# Run the complete benchmark process with required parameters
run-benchmark --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_tokens 50 --output_dir benchmark_results
```

This script will:
1. Run the benchmark comparing the original and quantized models
2. Save the results to a JSON file
3. Generate a visual HTML report
4. Open the report (on macOS) or provide the path to the report

### Using the Core Benchmarking Tool Directly

For more control over the benchmarking process, you can use `benchmark-model` directly:

```bash
# Basic usage - compare original and quantized models
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cpu

# Specify custom prompts
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --prompts-file custom_prompts.json

# Save results to a file
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --output benchmark_results.json
```

## Benchmark Metrics

The benchmarking tools collect and report several important metrics:

### Memory Usage

- Initial memory: Memory usage before loading the model
- Min memory: Minimum memory usage during the benchmark
- Max memory: Maximum memory usage during the benchmark
- Avg memory: Average memory usage during the benchmark
- Memory increase: Additional memory used after loading the model

### Loading Performance

- Load time: Time taken to load the model

### Generation Performance

- Prompt tokens: Number of tokens in the input prompts
- Prompt eval time: Time spent processing input prompts
- Prompt tokens/sec: Rate of processing input tokens
- Generated tokens: Number of tokens generated
- Generation time: Time spent generating tokens
- Generation tokens/sec: Rate of generating output tokens

## Comparing Multiple Models

You can benchmark multiple models to compare different quantization methods and bit widths:

```bash
# First, benchmark the original model
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized microsoft/Phi-4-mini-instruct --output original_results.json

# Then benchmark a quantized model
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model-gptq-4bit --output gptq_4bit_results.json

# Compare two quantized models
benchmark-model --original ./quantized-model-gptq-8bit --quantized ./quantized-model-gptq-4bit --output comparison_results.json
```

## Visualizing Benchmark Results

After running benchmarks, you can visualize the results:

```bash
# Generate visual report from benchmark results
visualize-benchmark --input benchmark_results.json --output_dir benchmark_report

# Open the HTML report
open benchmark_report/benchmark_report.html
```

The visualization script generates:
- An HTML report with detailed metrics
- Charts comparing memory usage
- Charts comparing performance metrics
- Charts comparing performance by prompt category

## Advanced Benchmarking Options

### Customizing Generation Parameters

You can customize the generation parameters used during benchmarking:

```bash
# Adjust generation parameters
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --max-new-tokens 100 --temperature 0.7 --top-p 0.9
```

### Batch Processing

For models that support it, you can enable batch processing:

```bash
# Enable batch processing
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --batch-size 4
```

### Custom Prompts

You can provide custom prompts for benchmarking:

```json
// custom_prompts.json
[
  {
    "category": "short_factual",
    "text": "What is the capital of France?"
  },
  {
    "category": "medium_creative",
    "text": "Write a short poem about artificial intelligence."
  },
  {
    "category": "long_reasoning",
    "text": "Explain the theory of relativity in detail, covering both special and general relativity. Include the key equations and their implications."
  }
]
```

```bash
# Use custom prompts
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --prompts-file custom_prompts.json
```

## Interpreting Benchmark Results

When interpreting benchmark results, consider the following:

### Memory Usage

- **Memory reduction**: Quantized models should show significant memory reduction compared to the original model
- **Expected reduction**: 
  - 8-bit quantization: ~50% reduction
  - 4-bit quantization: ~75% reduction
- **Actual vs. theoretical**: Actual memory usage may differ from theoretical calculations due to implementation details

### Speed

- **Loading time**: Quantized models may load faster or slower than the original model
- **Inference speed**: Quantized models often have different inference speeds:
  - 4-bit models are sometimes faster than 8-bit models due to reduced memory bandwidth requirements
  - CPU vs. GPU performance can vary significantly

### Quality

- **Output quality**: Quantized models may show slight degradation in output quality
- **Perplexity**: Higher perplexity indicates lower quality
- **Human evaluation**: Always perform human evaluation of outputs for critical use cases

## Device-Specific Benchmarking

You can benchmark on different devices to understand performance characteristics:

```bash
# Benchmark on CPU
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cpu

# Benchmark on CUDA (if available)
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cuda

# Benchmark on MPS (Apple Silicon)
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device mps
```

## Stress Testing

For critical applications, consider stress testing with complex prompts and longer outputs:

```bash
# Stress test with complex prompts and longer outputs
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --max-new-tokens 1024 --prompts-file complex_prompts.json
```

## Saving and Sharing Benchmark Results

You can save benchmark results to share with others or include in your model card:

```bash
# Save benchmark results with a timestamp
DATE=$(date +"%Y%m%d_%H%M%S")
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --output benchmark_$DATE.json
```

## Conclusion

Benchmarking is an essential step in the model quantization process. It helps you understand the trade-offs between different quantization methods and bit widths, and make informed decisions about which quantized model to use for your specific use case.

Always benchmark your quantized models before publishing them to ensure they meet your performance and quality requirements. 