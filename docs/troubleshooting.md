# Troubleshooting Guide

This guide provides solutions for common issues encountered when using the Model Quantizer tool.

## Platform-Specific Issues

### macOS Issues

#### BitsAndBytes on macOS

BitsAndBytes quantization is not fully supported on macOS. If you encounter issues, consider using GPTQ quantization instead:

```bash
# Use GPTQ instead of BitsAndBytes on macOS
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --bits 4
```

The Model Quantizer will automatically detect macOS and warn you if you try to use BitsAndBytes. It will also offer to switch to GPTQ automatically.

#### MPS Acceleration on Apple Silicon

When using MPS (Metal Performance Shaders) acceleration on Apple Silicon (M1/M2/M3), you may encounter operations that are not supported. The Model Quantizer automatically sets the `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable to handle this, but you can also set it manually:

```bash
# Set MPS fallback environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Then run the quantizer
model-quantizer microsoft/Phi-4-mini-instruct --device mps
```

#### Slow Quantization on macOS

Quantization on macOS may be slower than on Linux or Windows with CUDA. This is expected behavior due to the lack of optimized kernels for some operations. For faster quantization, consider using a machine with CUDA support.

### Windows Issues

#### CUDA Installation

If you encounter CUDA-related errors on Windows, ensure that you have the correct CUDA version installed for your PyTorch version. You can check the compatible CUDA version on the [PyTorch website](https://pytorch.org/get-started/locally/).

#### Path Length Limitations

Windows has path length limitations that can cause issues when saving models with long names. Use shorter output directory names or enable long path support in Windows:

```bash
# Use a shorter output directory name
model-quantizer microsoft/Phi-4-mini-instruct --output-dir phi4-mini
```

### Linux Issues

#### CUDA Out of Memory

If you encounter CUDA out of memory errors on Linux, try reducing the batch size or using a smaller model:

```bash
# Set a smaller batch size for calibration
model-quantizer microsoft/Phi-4-mini-instruct --additional-params '{"batch_size": 1}'
```

## Method-Specific Issues

### GPTQ Issues

#### GPTQ Calibration Dataset

If you encounter issues with the default calibration dataset, try providing your own:

```bash
# Use a custom calibration dataset
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --calibration-dataset "This is a sample text,This is another sample"
```

#### GPTQ Memory Usage

GPTQ quantization can use a significant amount of memory during the quantization process. If you encounter memory issues, try reducing the group size:

```bash
# Use a smaller group size
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --group-size 64
```

### BitsAndBytes Issues

#### BitsAndBytes Compatibility

BitsAndBytes quantization is primarily designed for CUDA devices and may not work well on CPU or MPS. If you encounter issues, consider using GPTQ quantization instead.

#### BitsAndBytes Installation

If you encounter issues installing BitsAndBytes, try installing it manually:

```bash
pip install bitsandbytes
```

### AWQ Issues

#### AWQ Installation

If you encounter issues installing AWQ, try installing it manually:

```bash
pip install autoawq
```

#### AWQ Compatibility

AWQ requires specific hardware support and may not work on all devices. Make sure your hardware is compatible before using AWQ quantization.

## General Issues

### Import Errors

If you encounter import errors, make sure you have installed all the required dependencies:

```bash
pip install -e .
```

### Memory Issues

If you encounter memory issues during quantization, try the following:

1. Use a smaller model
2. Use a smaller calibration dataset
3. Use a smaller group size
4. Use a lower bit width (e.g., 4-bit instead of 8-bit)
5. Use a device with more memory

### Slow Quantization

If quantization is slow, try the following:

1. Use a faster device (e.g., CUDA instead of CPU)
2. Use a smaller calibration dataset
3. Use a smaller model

### Model Loading Issues

If you encounter issues loading the quantized model, make sure you have installed all the required dependencies and are using the correct device:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained("MODEL_NAME-quantized", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME-quantized")
```

### Publishing Issues

If you encounter issues publishing to Hugging Face Hub, make sure you have logged in:

```bash
huggingface-cli login
```

## Getting Help

If you encounter an issue not covered in this guide, please open an issue on the [GitHub repository](https://github.com/lpalbou/model-quantizer/issues). 