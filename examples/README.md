# Model Quantizer Examples

This directory contains examples demonstrating how to use the Model Quantizer tool.

## Available Examples

- `quantize_example.py`: A script for quantizing any Hugging Face model
- `custom_prompts.json`: Example prompts for benchmarking and testing

## Basic Usage

```bash
# Quantize a model using GPTQ 4-bit
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --output-dir ./phi4-mini-gptq-4bit

# Quantize a model using BitsAndBytes 8-bit
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method bnb --bits 8 --output-dir ./phi4-mini-bnb-8bit

# Quantize a model and publish to Hugging Face Hub
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --publish --repo-id YOUR_USERNAME/phi4-mini-gptq-4bit
```

## Complete Workflow

### 1. Quantize a Model

```bash
# Quantize Gemma 2B to 4-bit using GPTQ
python quantize_example.py --model google/gemma-2b --method gptq --bits 4 --output-dir ./gemma-2b-quantized
```

### 2. Benchmark the Quantized Model

```bash
# Run the automated benchmark process
run-benchmark --original google/gemma-2b --quantized ./gemma-2b-quantized --device cpu --max_tokens 50 --output_dir benchmark_results
```

### 3. Test Interactively

```bash
# Chat with the model
chat-with-model --model_path ./gemma-2b-quantized --device cpu
```

### 4. Publish to Hugging Face Hub

```bash
# Publish the quantized model
python quantize_example.py --model google/gemma-2b --method gptq --bits 4 --publish --repo-id YOUR_USERNAME/gemma-2b-gptq-4bit
```

## Advanced Options

### Custom Calibration Dataset

```bash
# Use a custom calibration dataset
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --calibration-dataset "This is a sample text,This is another sample"
```

### Device Selection

```bash
# Use CPU for quantization
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --device cpu

# Use CUDA for quantization
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --device cuda

# Use MPS for quantization (Apple Silicon)
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --device mps
```

### Quantization Parameters

```bash
# Set group size to 64
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --group-size 64

# Use descending activation order
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --desc-act

# Use asymmetric quantization
python quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --no-sym
``` 