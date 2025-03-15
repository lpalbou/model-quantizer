# Troubleshooting Guide

This guide provides solutions for common issues encountered when using the Model Quantizer.

## Dependency Management

### Complete Installation (v0.3.1+)

As of version 0.3.1, we provide installation scripts that ensure all dependencies, including GPTQ support for optimum, are properly installed:

```bash
# For Linux/macOS:
chmod +x install_dependencies.sh
./install_dependencies.sh

# For Windows:
install_dependencies.bat
```

These scripts handle the correct installation order and ensure that optimum is installed with GPTQ support.

### Using requirements-all.txt (v0.3.1+)

If you prefer to use the requirements file directly, be aware that some packages like gptqmodel require torch to be installed first, and optimum needs to be installed with GPTQ support:

```bash
# First install torch and related packages
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Then install optimum with GPTQ support
pip install "optimum[gptq]==1.24.0"

# Then install the rest of the dependencies
pip install -r requirements-all.txt
```

### Complete Installation (v0.3.0)

As of version 0.3.0, we provide installation scripts for a reliable one-shot installation of all dependencies:

```bash
# For Linux/macOS:
chmod +x install_dependencies.sh
./install_dependencies.sh

# For Windows:
install_dependencies.bat
```

These scripts handle the correct installation order, ensuring that torch is installed before gptqmodel and other dependencies that require it.

### Complete Installation (v0.2.9)

For version 0.2.9, we provide a `requirements-all.txt` file for a one-shot installation of all dependencies:

```bash
# Install all dependencies at once
pip install -r requirements-all.txt
```

This installs everything needed for all features, including:
- Core dependencies (transformers, huggingface_hub)
- PyTorch with its components
- All quantization methods (GPTQ, BitsAndBytes, AWQ)
- Visualization tools
- Data handling utilities
- Development tools

### Minimal Installation (v0.2.8+)

If you prefer a minimal installation, we still support the modular approach:

```bash
# Install only core dependencies
pip install model-quantizer

# Then add specific features as needed
pip install model-quantizer[gptq]  # For GPTQ support
pip install model-quantizer[viz]   # For visualization
```

#### Installing Optional Features

```bash
# Install with GPTQ support (includes optimum and gptqmodel)
pip install model-quantizer[gptq]

# Install with visualization support
pip install model-quantizer[viz]

# Install with dataset handling support
pip install model-quantizer[data]

# Install with all features
pip install model-quantizer[all]
```

#### Installing PyTorch

PyTorch is not a direct dependency to allow more flexible installation. Install it separately:

```bash
# Install PyTorch (CPU only)
pip install torch

# Or install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for the correct command for your system
```

### Version Compatibility

If you encounter compatibility issues, these specific versions are known to work well together:

```bash
# Core dependencies
pip install transformers>=4.30.0 huggingface_hub>=0.16.0 numpy==1.26.4 psutil==7.0.0 tqdm==4.67.1

# PyTorch (install separately)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# GPTQ dependencies
pip install optimum==1.24.0 gptqmodel<2.1.0

# Other quantization methods (install only what you need)
pip install bitsandbytes==0.42.0  # For BitsAndBytes
pip install autoawq  # For AWQ
```

## Platform-Specific Information

### macOS Compatibility (v0.3.1+)

Model Quantizer has been successfully tested on macOS Sonoma 14.2 with both Python 3.11 and Python 3.12. The installation scripts ensure proper dependency installation for both Python versions.

Key points for macOS users:
- Use the installation script for the most reliable setup
- Both Python 3.11 and 3.12 are fully supported
- For GPTQ quantization, ensure you have `optimum[gptq]` installed, not just `optimum`
- When quantizing models, explicitly specify `--device cpu` for macOS

Example command for macOS users:
```bash
python -m quantizer.cli microsoft/Phi-4-mini-instruct --bits 4 --method gptq --output-dir qmodels/phi4-mini-4bit --device cpu
```

### Windows Compatibility

Model Quantizer works on Windows with both Python 3.11 and 3.12. Use the `install_dependencies.bat` script for proper installation.

### Linux Compatibility

Model Quantizer works on Linux with both Python 3.11 and 3.12. Use the `install_dependencies.sh` script for proper installation.

## Table of Contents
- [GPTQ Quantization Issues](#gptq-quantization-issues)
- [Python Version Compatibility](#python-version-compatibility)
- [Memory Issues](#memory-issues)
- [Slow Generation](#slow-generation)
- [Model Loading Errors](#model-loading-errors)
- [Compatibility Issues](#compatibility-issues)

## Platform-Specific Issues

### macOS Issues

#### BitsAndBytes on macOS

BitsAndBytes quantization is not fully supported on macOS. If you encounter issues, consider using GPTQ quantization instead:

```bash
# Use GPTQ instead of BitsAndBytes on macOS
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --bits 4 --device cpu
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

## Python Version Compatibility

### Python 3.12 and GPTQ

If you're using Python 3.12 and encounter this error with GPTQ quantization:

```
ERROR:root:Error quantizing model: GPU is required to quantize or run quantize model.
```

This is due to a compatibility issue between transformers 4.49.0 and Python 3.12. Model Quantizer version 0.2.4+ includes a completely redesigned fix for this issue. Make sure you're using the latest version:

```bash
pip install -U model-quantizer
```

The fix in version 0.2.4 uses multiple advanced patching strategies:
- Patches the GPTQConfig.post_init method to skip CUDA checks on CPU
- Recursively scans all transformers modules to find and patch functions with CUDA checks
- Adds method-level exception handling to bypass GPU requirements when running on CPU
- Works with transformers 4.49.0 and Python 3.12.7

If you still encounter issues, you can try:

1. **Explicitly specify the CPU device**:
   ```bash
   model-quantizer your-model --method gptq --device cpu
   ```

2. **Use Python 3.11 instead**:
   ```bash
   # Create a Python 3.11 environment
   conda create -n py311 python=3.11
   conda activate py311
   pip install model-quantizer
   ```

3. **Use BitsAndBytes instead**:
   ```bash
   model-quantizer your-model --method bitsandbytes
   ```

## Method-Specific Issues

### GPTQ Quantization Issues

#### GPTQ Device Selection

GPTQ quantization works on both CPU and GPU devices. If you encounter issues with one device, try the other:

```bash
# Try CPU if GPU is causing issues
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --device cpu

# Try GPU if available and CPU is slow
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --device cuda
```

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

#### Missing GPTQ Dependencies

**Problem**: When attempting to use GPTQ quantization, you may encounter errors related to missing dependencies.

**Solution**: As of version 0.2.8, we've moved optimum to the gptq extras to make dependencies clearer:

```bash
# Install the package with GPTQ extras (includes both optimum and gptqmodel)
pip install model-quantizer[gptq]

# Or install dependencies directly
pip install optimum==1.24.0 gptqmodel<2.1.0
```

If you're using Python 3.12, note that some quantization libraries may not be fully compatible yet. We recommend using Python 3.11 for the most reliable experience with GPTQ quantization.

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

### GPTQ Installation Issues (v0.3.1+)

**Problem 1**: When installing from `requirements-all.txt`, you may encounter an error with gptqmodel installation:

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**: This occurs because gptqmodel requires torch to be installed first. As of version 0.3.1, we provide solutions:

1. **Use the installation scripts** (recommended):
   ```bash
   # For Linux/macOS:
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   
   # For Windows:
   install_dependencies.bat
   ```

2. **Install torch first, then other dependencies**:
   ```bash
   # First install torch and related packages
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
   
   # Then install the rest of the dependencies
   pip install -r requirements-all.txt
   ```

3. **Install gptqmodel separately after torch**:
   ```bash
   pip install torch==2.5.1
   pip install "gptqmodel<2.1.0"
   ```

**Problem 2**: When using GPTQ quantization, you may encounter an error even though optimum is installed:

```
ERROR: Loading a GPTQ quantized model requires optimum (`pip install optimum`)
```

**Solution**: This occurs because optimum needs to be installed with GPTQ support:

1. **Install optimum with GPTQ support**:
   ```bash
   pip install "optimum[gptq]"
   ```

2. **Use the installation scripts** (recommended):
   ```bash
   # For Linux/macOS:
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   
   # For Windows:
   install_dependencies.bat
   ```

3. **If you're still having issues**, try installing the specific version of gptqmodel mentioned in the error:
   ```bash
   pip install gptqmodel>=2.1.0
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

### Python 3.12 Compatibility Issues

**Problem**: When using Python 3.12, you may encounter errors related to CUDA availability checks in the transformers library, even when specifying `--device cpu`.

**Solution**: 

In version 0.2.8, we've further refined the dependency structure and recommend using Python 3.11 for GPTQ quantization:

```bash
# Create a Python 3.11 environment
conda create -n py311 python=3.11
conda activate py311

# Install the package with GPTQ support
pip install model-quantizer[gptq]
```

If you must use Python 3.12, try:

1. Installing specific versions of dependencies:
   ```bash
   pip install torch==2.5.1 optimum==1.24.0 gptqmodel<2.1.0 numpy==1.26.4
   ```

2. Using a different quantization method:
   ```bash
   model-quantizer your-model --method bitsandbytes
   ```

## Memory Issues

### Issue

The model fails to load with an out-of-memory error.

### Solutions

1. **Use a Lower Bit Width**

   4-bit quantization uses significantly less memory than 8-bit:

   ```bash
   model-quantizer your-model --bits 4 --method gptq
   ```

2. **Use CPU Instead of GPU**

   If your GPU has limited memory, try using CPU:

   ```bash
   model-quantizer your-model --device cpu
   ```

3. **Reduce Model Size**

   Consider using a smaller model if the current one is too large for your hardware.

## Slow Generation

### Issue

Text generation is very slow, especially on CPU.

### Solutions

1. **Use a GPU if Available**

   ```bash
   model-quantizer your-model --device cuda  # or mps for Mac
   ```

2. **Reduce Generation Parameters**

   When using the chat interface, reduce the maximum tokens:

   ```bash
   chat-with-model --model_path ./quantized-model --max_new_tokens 100
   ```

3. **Use a Smaller Model**

   Smaller models (1-3B parameters) generate text much faster than larger ones.

## Model Loading Errors

### Issue

Errors when loading a quantized model.

### Solutions

1. **Check Compatibility**

   Ensure you're using compatible versions of PyTorch, Transformers, and Optimum.

2. **Verify Model Format**

   Make sure the model was quantized with the correct method for your hardware.

3. **Clear Cache**

   Try clearing the Hugging Face cache:

   ```bash
   rm -rf ~/.cache/huggingface/
   ```

## Compatibility Issues

### Issue

BitsAndBytes doesn't work on macOS/OSX.

### Solutions

1. **Use GPTQ Instead**

   ```bash
   model-quantizer your-model --method gptq --device cpu
   ```

2. **Use a Different Quantization Method**

   Try AWQ if your model supports it:

   ```bash
   model-quantizer your-model --method awq
   ```

## Getting Help

If you encounter an issue not covered in this guide, please open an issue on the [GitHub repository](https://github.com/lpalbou/model-quantizer/issues). 

### ImportError or ModuleNotFoundError

**Problem**: You encounter an `ImportError` or `ModuleNotFoundError` when running the quantizer.

**Solution**: Make sure you have installed all the required dependencies with the exact versions. As of version 0.2.6, we provide a comprehensive `requirements.txt` file with pinned versions:

```bash
pip install -r requirements.txt
```

Or install the package with all extras:

```bash
pip install -e ".[all]"
```

### CUDA Out of Memory Errors

**Problem**: You encounter CUDA out of memory errors when quantizing large models.

**Solution**: Try the following:

1. Reduce the batch size using the `--batch-size` parameter
2. Use a smaller model or a model with fewer parameters
3. Try a different quantization method that requires less memory
4. If available, use a GPU with more VRAM

### Slow Quantization

**Problem**: The quantization process is taking a very long time.

**Solution**: 

1. Use a GPU if available, as CPU quantization is significantly slower
2. Reduce the number of examples used for calibration using the `--calib-examples` parameter
3. Try a different quantization method that may be faster for your specific model

### Model Saving Issues

**Problem**: You encounter errors when saving the quantized model.

**Solution**: 

1. Ensure you have write permissions to the output directory
2. Check if you have enough disk space
3. Try using a different output directory
4. If the error persists, check the specific error message for more details

## Still Having Issues?

If you're still experiencing problems after trying these solutions, please open an issue on our GitHub repository with the following information:

1. The exact command you're running
2. The complete error message
3. Your Python version (`python --version`)
4. Your operating system
5. The versions of key dependencies:
   ```bash
   pip list | grep -E "transformers|torch|optimum|gptqmodel|autoawq|bitsandbytes"
   ```

This will help us diagnose and fix the issue more quickly. 

## Dependency Version Issues

### Version Conflicts

**Problem**: You encounter errors related to incompatible package versions.

**Solution**: As of version 0.2.8, we've further refined the dependency structure to minimize conflicts:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install with specific extras
pip install model-quantizer[gptq]  # For GPTQ support
pip install model-quantizer[bitsandbytes]  # For BitsAndBytes support
```

### NumPy Version Issues

**Problem**: You encounter errors related to NumPy version compatibility, especially with gptqmodel 2.1.0+ requiring numpy>=2.2.2.

**Solution**: We use gptqmodel versions below 2.1.0 to avoid requiring numpy>=2.2.2:

```bash
# Install numpy 1.26.4 which is compatible with our dependencies
pip install numpy==1.26.4

# Install gptqmodel < 2.1.0
pip install gptqmodel<2.1.0
```

If you're still experiencing issues, try:

```bash
# Uninstall numpy first
pip uninstall -y numpy

# Then install the specific version
pip install numpy==1.26.4
```

## AWQ Quantization Issues

### Missing AWQ Dependencies

**Problem**: When attempting to use AWQ quantization, you may encounter errors related to missing dependencies.

**Solution**: Install the required AWQ dependencies:

```bash
pip install model-quantizer[awq]

# Or install autoawq directly
pip install autoawq
```

## BitsAndBytes Quantization Issues

### Missing BitsAndBytes Dependencies

**Problem**: When attempting to use BitsAndBytes quantization, you may encounter errors related to missing dependencies.

**Solution**: Install the required BitsAndBytes dependencies:

```bash
pip install model-quantizer[bitsandbytes]

# Or install bitsandbytes directly
pip install bitsandbytes==0.42.0
``` 