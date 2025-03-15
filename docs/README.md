# Model Quantizer Documentation

This repository contains comprehensive documentation for the Model Quantizer tool, which allows you to quantize Hugging Face models to reduce their memory footprint while maintaining most of their performance.

## Core Documentation

- [General Guide](general_guide.md): Comprehensive guide to quantizing Hugging Face models
- [Benchmarking Guide](benchmarking.md): How to benchmark and compare quantized models
- [Publishing Guide](publishing.md): How to publish quantized models to Hugging Face Hub, including automatic model card generation
- [Chat Guide](chat_guide.md): How to interactively test your quantized models
- [Troubleshooting Guide](troubleshooting.md): Solutions to common issues encountered during quantization

## Example-Specific Documentation

- [Phi-4-mini Quantization Guide](phi4_mini.md): Detailed guide for quantizing the Microsoft Phi-4-mini model

## Core Workflow

The Model Quantizer provides a complete workflow for working with quantized models:

### 1. Quantize

First, quantize your model to reduce its memory footprint:

```bash
model-quantizer MODEL_NAME --bits 4 --method gptq --output-dir quantized-model
```

See the [General Model Quantization Guide](general_guide.md) for detailed instructions.

### 2. Benchmark

Next, benchmark your quantized model to evaluate its performance:

```bash
run-benchmark --original MODEL_NAME --quantized ./quantized-model --device cpu
```

See the [Benchmarking Guide](benchmarking.md) for detailed instructions.

### 3. Test Interactively

Test your quantized model interactively to verify its quality:

```bash
chat-with-model --model_path ./quantized-model
```

### 4. Publish

Finally, publish your quantized model to the Hugging Face Hub:

```bash
model-quantizer MODEL_NAME --bits 4 --method gptq --output-dir quantized-model --publish --repo-id YOUR_USERNAME/MODEL_NAME-gptq-4bit
```

See the [Publishing Guide](publishing_guide.md) for detailed instructions.

## Getting Started

If you're new to model quantization, we recommend starting with the [General Model Quantization Guide](general_guide.md), which provides an overview of the quantization process and available methods.

For specific models, check if there's a dedicated guide (like the [Phi-4-Mini Quantization Guide](phi4_mini.md)) that provides optimized settings and recommendations.

## Examples

For practical examples, see the [examples](../examples) directory, which contains scripts for:

- Quantizing models
- Using quantized models
- Benchmarking performance
- Visualizing results
- Comparing memory usage

## Contributing

If you'd like to contribute to the documentation:

1. Fork the repository
2. Create a new branch for your changes
3. Add or update documentation files
4. Submit a pull request

We welcome improvements to existing guides and new model-specific guides.

## Support

If you encounter issues not covered in the [Troubleshooting Guide](troubleshooting.md), please open an issue on the GitHub repository. 