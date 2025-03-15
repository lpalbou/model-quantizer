# Benchmarking Quantized Models

This guide explains how to benchmark quantized models to evaluate their performance, memory usage, and output quality compared to the original models.

## Why Benchmark?

Benchmarking is essential to understand the trade-offs between:
- Memory efficiency
- Inference speed
- Output quality

Different quantization methods and bit widths will have varying impacts on these factors, and benchmarking helps you make informed decisions.

## Benchmarking Tools

The Model Quantizer package includes several tools for benchmarking:

1. `benchmark_phi4_mini.py`: A script for benchmarking the Phi-4-mini model
2. `visualize_benchmark.py`: A script for visualizing benchmark results
3. `compare_memory_usage.py`: A script for comparing memory usage

These scripts can be adapted for other models as well.

## Performance Metrics

When benchmarking quantized models, consider the following metrics:

### 1. Inference Speed

Measure the time taken to generate text:

```bash
# Example command to benchmark inference speed
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --metric speed
```

This will measure:
- Tokens per second
- Time per token
- Total generation time

### 2. Memory Usage

Measure the memory footprint of the model:

```bash
# Example command to benchmark memory usage
python examples/compare_memory_usage.py --model-paths MODEL_PATH_1 MODEL_PATH_2 --device cuda
```

This will measure:
- Peak memory usage
- Memory usage during loading
- Memory usage during inference

### 3. Output Quality

Evaluate the quality of generated text:

```bash
# Example command to benchmark output quality
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --metric quality --prompts-file custom_prompts.json
```

This will evaluate:
- Perplexity on a test dataset
- ROUGE scores compared to reference outputs
- BLEU scores compared to reference outputs

## Setting Up a Benchmark

### 1. Prepare Models

Quantize your model using different methods and bit widths:

```bash
# Quantize with GPTQ at 8-bit
model-quantizer MODEL_NAME --bits 8 --method gptq --output-dir MODEL_NAME-gptq-8bit

# Quantize with GPTQ at 4-bit
model-quantizer MODEL_NAME --bits 4 --method gptq --output-dir MODEL_NAME-gptq-4bit

# Quantize with BitsAndBytes at 8-bit
model-quantizer MODEL_NAME --bits 8 --method bitsandbytes --output-dir MODEL_NAME-bnb-8bit

# Quantize with BitsAndBytes at 4-bit
model-quantizer MODEL_NAME --bits 4 --method bitsandbytes --output-dir MODEL_NAME-bnb-4bit
```

### 2. Prepare Test Data

Create a file with test prompts:

```json
[
  {
    "prompt": "Explain the concept of quantum computing in simple terms.",
    "reference": "Quantum computing uses quantum bits or qubits that can be both 0 and 1 at the same time, unlike classical bits. This allows quantum computers to process certain types of problems much faster than traditional computers."
  },
  {
    "prompt": "Write a short poem about artificial intelligence.",
    "reference": "Silicon dreams and neural streams,\nLearning patterns, crafting schemes.\nMind of math, soul of code,\nWalking down a human road."
  }
]
```

Save this as `custom_prompts.json`.

### 3. Run Benchmarks

Run benchmarks for each model:

```bash
# Benchmark original model
python examples/benchmark_phi4_mini.py --model-path microsoft/Phi-4-mini-instruct --output-file results/original.json

# Benchmark GPTQ 8-bit
python examples/benchmark_phi4_mini.py --model-path MODEL_NAME-gptq-8bit --output-file results/gptq-8bit.json

# Benchmark GPTQ 4-bit
python examples/benchmark_phi4_mini.py --model-path MODEL_NAME-gptq-4bit --output-file results/gptq-4bit.json

# Benchmark BitsAndBytes 8-bit
python examples/benchmark_phi4_mini.py --model-path MODEL_NAME-bnb-8bit --output-file results/bnb-8bit.json

# Benchmark BitsAndBytes 4-bit
python examples/benchmark_phi4_mini.py --model-path MODEL_NAME-bnb-4bit --output-file results/bnb-4bit.json
```

### 4. Visualize Results

Visualize the benchmark results:

```bash
# Visualize speed comparison
python examples/visualize_benchmark.py --results-files results/*.json --metric speed --output-file results/speed_comparison.png

# Visualize quality comparison
python examples/visualize_benchmark.py --results-files results/*.json --metric quality --output-file results/quality_comparison.png
```

## Customizing Benchmarks

### Custom Metrics

You can add custom metrics to your benchmarks:

```python
def custom_metric(generated_text, reference_text):
    # Implement your custom metric
    return score

# Add to benchmark script
metrics["custom"] = custom_metric
```

### Custom Generation Parameters

Adjust generation parameters to match your use case:

```bash
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --max-new-tokens 100 --temperature 0.7 --top-p 0.9
```

### Batch Processing

For faster benchmarking, use batch processing:

```bash
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --batch-size 4
```

## Interpreting Results

### Speed vs. Quality Trade-off

- Lower bit widths generally provide faster inference but may reduce quality
- Different quantization methods may have different speed-quality trade-offs
- Consider your specific use case when interpreting results

### Memory Efficiency

- Lower bit widths significantly reduce memory usage
- Some methods may have overhead that reduces memory savings
- Consider both loading and inference memory usage

### Hardware Considerations

- Results may vary significantly across different hardware
- CUDA devices may show different patterns than CPU or MPS
- Always benchmark on the target hardware

## Example Benchmark Report

Here's an example of how to structure a benchmark report:

```markdown
# Benchmark Report: MODEL_NAME

## Test Environment
- Hardware: NVIDIA RTX 3090 / Apple M1 Pro / Intel i9-12900K
- Operating System: Ubuntu 22.04 / macOS 13.0 / Windows 11
- PyTorch Version: 2.0.1
- CUDA Version: 11.7

## Models Tested
- Original (FP16)
- GPTQ 8-bit
- GPTQ 4-bit
- BitsAndBytes 8-bit
- BitsAndBytes 4-bit

## Speed Benchmark
| Model | Tokens/Second | Time/Token (ms) | Total Time (s) |
| ----- | ------------- | --------------- | -------------- |
| Original | 10.5 | 95.2 | 9.52 |
| GPTQ 8-bit | 12.3 | 81.3 | 8.13 |
| GPTQ 4-bit | 15.7 | 63.7 | 6.37 |
| BnB 8-bit | 11.8 | 84.7 | 8.47 |
| BnB 4-bit | 14.2 | 70.4 | 7.04 |

## Memory Usage
| Model | Loading (GB) | Inference (GB) | Peak (GB) |
| ----- | ------------ | -------------- | --------- |
| Original | 7.6 | 7.8 | 8.1 |
| GPTQ 8-bit | 3.8 | 4.0 | 4.2 |
| GPTQ 4-bit | 1.9 | 2.1 | 2.3 |
| BnB 8-bit | 3.8 | 4.1 | 4.3 |
| BnB 4-bit | 1.9 | 2.2 | 2.4 |

## Quality Metrics
| Model | Perplexity | ROUGE-L | BLEU |
| ----- | ---------- | ------- | ---- |
| Original | 3.21 | 0.85 | 0.72 |
| GPTQ 8-bit | 3.25 | 0.84 | 0.71 |
| GPTQ 4-bit | 3.42 | 0.81 | 0.68 |
| BnB 8-bit | 3.27 | 0.83 | 0.70 |
| BnB 4-bit | 3.45 | 0.80 | 0.67 |

## Conclusion
- GPTQ 4-bit provides the best balance of speed, memory efficiency, and quality
- BitsAndBytes 8-bit is recommended for CUDA devices where quality is important
- Original model is only recommended when memory is not a constraint
```

## Advanced Benchmarking

### Multi-Device Testing

Test across different devices:

```bash
# Test on CPU
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --device cpu

# Test on CUDA
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --device cuda

# Test on MPS (Apple Silicon)
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --device mps
```

### Stress Testing

Test with longer sequences and more complex prompts:

```bash
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --max-new-tokens 1024 --prompts-file complex_prompts.json
```

### Continuous Benchmarking

Set up continuous benchmarking to track performance over time:

```bash
# Create a benchmark script
#!/bin/bash
DATE=$(date +%Y-%m-%d)
python examples/benchmark_phi4_mini.py --model-path MODEL_PATH --output-file results/benchmark_$DATE.json
```

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Model Quantizer Documentation](https://github.com/lpalbou/model-quantizer) 