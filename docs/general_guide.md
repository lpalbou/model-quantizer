# General Model Quantization Guide

This guide explains how to quantize any Hugging Face model using the Model Quantizer tool.

## Why Use Model Quantizer?

Model Quantizer addresses several key challenges in using quantized models:

1. **Cross-Platform Compatibility**: The BitsAndBytes library, while powerful, doesn't work on macOS/OSX. Model Quantizer provides GPTQ quantization that works across all platforms including macOS.

2. **Time-Consuming Quantization**: Quantizing models with GPTQ takes significant time, especially for larger models. By providing tools to publish pre-quantized models, Model Quantizer allows users to quantize once and reuse many times.

3. **Control Over Quantization**: Unlike other published quantized models, this tool gives you full control over the quantization process, including bit width, quantization method, and calibration dataset.

4. **Easy Publishing**: Model Quantizer streamlines the process of publishing quantized models to the Hugging Face Hub, making them accessible to the community.

## Introduction to Model Quantization

Model quantization is a technique to reduce the memory footprint and computational requirements of large language models (LLMs) while maintaining most of their performance. This is achieved by representing the model weights with fewer bits than the standard 32-bit or 16-bit floating-point precision.

The Model Quantizer tool integrates with Hugging Face's quantization methods, making it easy to quantize and publish models for reuse.

## Supported Quantization Methods

### GPTQ Quantization

GPTQ is a post-training quantization technique implemented by Hugging Face where each row of the weight matrix is quantized independently to find a version of the weights that minimizes error. These weights are quantized to int4 or int8, stored as int32, and dequantized to fp16 on the fly during inference.

```bash
# 8-bit GPTQ quantization
model-quantizer MODEL_NAME --bits 8 --method gptq --output-dir MODEL_NAME-gptq-8bit

# 4-bit GPTQ quantization (recommended)
model-quantizer MODEL_NAME --bits 4 --method gptq --output-dir MODEL_NAME-gptq-4bit
```

**Note**: While the tool allows for 2-bit and 3-bit options with GPTQ, these are experimental and may not be fully supported by Hugging Face's implementation. 4-bit and 8-bit are the recommended and well-tested options.

### BitsAndBytes Quantization

BitsAndBytes is a library for quantizing models to 4 or 8 bits. It's particularly useful for models that will be run on CUDA devices.

```bash
# 8-bit BitsAndBytes quantization
model-quantizer MODEL_NAME --bits 8 --method bitsandbytes --output-dir MODEL_NAME-bnb-8bit

# 4-bit BitsAndBytes quantization
model-quantizer MODEL_NAME --bits 4 --method bitsandbytes --output-dir MODEL_NAME-bnb-4bit
```

**Note**: BitsAndBytes only supports 4-bit and 8-bit quantization. Attempting to use other bit widths will result in an error.

### AWQ Quantization

AWQ (Activation-aware Weight Quantization) is a technique that considers the activation patterns during quantization to preserve model quality.

```bash
# 4-bit AWQ quantization (recommended)
model-quantizer MODEL_NAME --bits 4 --method awq --output-dir MODEL_NAME-awq-4bit
```

**Note**: AWQ is primarily designed for 4-bit quantization. While other bit widths may work, they are not well-tested and may not provide optimal results.

## Bit Width Selection

The Model Quantizer supports different bit widths for quantization, but the recommended options vary by method:

- **GPTQ**: 4-bit and 8-bit (recommended), 2-bit and 3-bit (experimental)
- **BitsAndBytes**: 4-bit and 8-bit only
- **AWQ**: 4-bit (recommended)

The choice of bit width depends on your specific requirements:

- Use 4-bit quantization when memory efficiency is critical
- Use 8-bit quantization when quality is more important than memory efficiency

## Device Considerations

Different quantization methods perform differently on various hardware:

- **GPTQ** generally provides better performance on CPU and Apple Silicon (MPS) devices
- **BitsAndBytes** is optimized for CUDA devices and may not work well on CPU or MPS
- **AWQ** works best on CUDA devices with specific hardware support

You can specify the device for quantization:

```bash
# Use CPU for quantization
model-quantizer MODEL_NAME --device cpu

# Use CUDA for quantization
model-quantizer MODEL_NAME --device cuda

# Use MPS for quantization (Apple Silicon)
model-quantizer MODEL_NAME --device mps

# Let the tool choose the best device
model-quantizer MODEL_NAME --device auto
```

## Calibration Dataset

Quantization methods like GPTQ and AWQ require a calibration dataset to optimize the quantization process. You can provide your own dataset:

```bash
# Use a comma-separated list of strings
model-quantizer MODEL_NAME --calibration-dataset "This is a sample text,This is another sample"

# Use a dataset from the Hugging Face Hub
model-quantizer MODEL_NAME --calibration-dataset "c4"
```

If no calibration dataset is provided, a default dataset will be used.

## Advanced Configuration

The Model Quantizer provides several advanced configuration options:

- **Group Size**: Controls the granularity of quantization (default: 128)
- **Descending Activation Order**: Whether to use descending activation order (default: False)
- **Symmetric Quantization**: Whether to use symmetric quantization (default: True)

```bash
# Set group size to 64
model-quantizer MODEL_NAME --group-size 64

# Use descending activation order
model-quantizer MODEL_NAME --desc-act

# Use asymmetric quantization
model-quantizer MODEL_NAME --no-sym
```

## Loading Quantized Models

Once the model is quantized, it can be loaded using the Hugging Face Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained("MODEL_NAME-quantized", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME-quantized")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Memory Usage Comparison

Here's a general comparison of theoretical memory usage for different quantization methods:

| Quantization Method | Bits | Theoretical Memory Reduction |
| ------------------- | ---- | ---------------------------- |
| None (FP16)         | 16   | 1x (baseline)                |
| GPTQ                | 8    | 2x                           |
| GPTQ                | 4    | 4x                           |
| BitsAndBytes        | 8    | 2x                           |
| BitsAndBytes        | 4    | 4x                           |
| AWQ                 | 4    | 4x                           |

The actual memory usage depends on the specific model architecture, size, and implementation details. We recommend benchmarking on your specific hardware and use case.

## Performance Considerations

- Lower bit widths (4-bit) provide better memory efficiency but may have slightly lower quality
- 8-bit quantization is a good balance between memory efficiency and quality
- Different quantization methods may perform differently on different models
- Always benchmark your specific model with different quantization methods and bit widths

## Troubleshooting

### GPTQ on macOS

If you encounter issues with GPTQ quantization on macOS, try setting the following environment variable:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This will allow PyTorch to fall back to CPU for operations not supported on MPS.

### BitsAndBytes on macOS

BitsAndBytes quantization may not work well on macOS as it's primarily designed for CUDA devices. If you're using macOS, we recommend using GPTQ quantization instead.

### AWQ Compatibility

AWQ requires specific hardware support and may not work on all devices. Make sure your hardware is compatible before using AWQ quantization.

## References

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [Hugging Face GPTQ Documentation](https://huggingface.co/docs/transformers/en/quantization/gptq)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Model Quantizer Documentation](https://github.com/lpalbou/model-quantizer) 