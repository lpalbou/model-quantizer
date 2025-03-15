# Quantizing the Microsoft Phi-4-mini-instruct Model

This guide provides detailed instructions for quantizing the Microsoft Phi-4-mini-instruct model using the Model Quantizer tool.

## Model Specifications

- **Model**: Microsoft Phi-4-mini-instruct
- **Size**: 4.2B parameters
- **Original Format**: 16-bit floating point (FP16)
- **Original Memory Usage**: ~8.4GB

## Why Quantize Phi-4-mini?

The Phi-4-mini model is a powerful yet relatively compact model that can run on consumer hardware. However, even at 4.2B parameters, it requires significant memory in its original form. Quantization offers several benefits:

1. **Reduced Memory Usage**: Quantized versions use significantly less memory
2. **Faster Loading**: Smaller models load faster
3. **Broader Accessibility**: Can run on devices with limited memory
4. **Comparable Performance**: Maintains most of the original model's capabilities

## Quantization Options

The Model Quantizer supports multiple quantization methods for Phi-4-mini:

### GPTQ Quantization (Recommended for most users)

GPTQ works well across all platforms, including macOS:

```bash
# 8-bit GPTQ (better quality, ~50% memory reduction)
model-quantizer microsoft/Phi-4-mini-instruct --bits 8 --method gptq --output-dir phi4-mini-gptq-8bit

# 4-bit GPTQ (better memory efficiency, ~75% memory reduction)
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq --output-dir phi4-mini-gptq-4bit
```

For Mac users, explicitly specifying the CPU device can help ensure compatibility:

```bash
# Explicitly use CPU device on Mac
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq --device cpu --output-dir phi4-mini-gptq-4bit
```

### BitsAndBytes Quantization (Best for CUDA devices)

BitsAndBytes is optimized for CUDA devices:

```bash
# 8-bit BitsAndBytes
model-quantizer microsoft/Phi-4-mini-instruct --bits 8 --method bitsandbytes --output-dir phi4-mini-bnb-8bit

# 4-bit BitsAndBytes
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method bitsandbytes --output-dir phi4-mini-bnb-4bit
```

### AWQ Quantization (Experimental)

AWQ can provide good results for 4-bit quantization:

```bash
# 4-bit AWQ
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method awq --output-dir phi4-mini-awq-4bit
```

## Benchmarking

After quantizing, benchmark the model to evaluate its performance:

```bash
# Run the automated benchmark process
run-benchmark --original microsoft/Phi-4-mini-instruct --quantized ./phi4-mini-gptq-4bit --device cpu --max_tokens 50 --output_dir benchmark_results --update-model-card
```

This will generate a comprehensive report comparing the original and quantized models and update the model card with benchmark results.

## Interactive Testing

Test the quantized model interactively:

```bash
# Chat with the model
chat-with-model --model_path ./phi4-mini-gptq-4bit --device cpu

# Use a custom system prompt
chat-with-model --model_path ./phi4-mini-gptq-4bit --system_prompt "You are a helpful AI assistant specialized in science."
```

## Publishing Your Quantized Model

Share your quantized model with the community:

```bash
# Publish to Hugging Face Hub
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq --output-dir phi4-mini-gptq-4bit --publish --repo-id YOUR_USERNAME/phi4-mini-gptq-4bit
```

## Performance Comparison

| Model Version | Memory Usage | Loading Time | Generation Speed | Quality |
|---------------|--------------|--------------|------------------|---------|
| Original (FP16) | ~8.4GB | Baseline | Baseline | Baseline |
| GPTQ 8-bit | ~4.2GB | Similar | Slightly slower | Very close to original |
| GPTQ 4-bit | ~2.1GB | Faster | Similar or faster | Slight degradation |
| BnB 8-bit | ~4.2GB | Similar | Similar | Very close to original |
| BnB 4-bit | ~2.1GB | Faster | Similar | Moderate degradation |

## Recommendations

- **For macOS users**: Use GPTQ 4-bit for best memory efficiency or GPTQ 8-bit for best quality
- **For Windows/Linux with CUDA**: Try both GPTQ and BitsAndBytes to see which performs better on your hardware
- **For memory-constrained devices**: Use 4-bit quantization (GPTQ recommended)
- **For quality-sensitive applications**: Use 8-bit quantization

## Troubleshooting

### GPTQ Device Selection

If you encounter issues with GPTQ quantization, try explicitly specifying the device:

```bash
# Try CPU device
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --bits 4 --device cpu

# Try CUDA device if available
model-quantizer microsoft/Phi-4-mini-instruct --method gptq --bits 4 --device cuda
```

### macOS-Specific Issues

If you encounter other issues on macOS, try:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This allows PyTorch to fall back to CPU for operations not supported on MPS.

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Try a lower bit width (4-bit instead of 8-bit)
2. Reduce the batch size for calibration
3. Use CPU for quantization instead of CUDA

### Slow Quantization

GPTQ quantization can be time-consuming. For faster results:

1. Use a smaller calibration dataset
2. Increase the group size parameter
3. Use a more powerful GPU if available

## Conclusion

The Phi-4-mini model is an excellent candidate for quantization, offering significant memory savings while maintaining most of its capabilities. The 4-bit GPTQ quantized version is particularly impressive, reducing memory usage by approximately 75% while still providing good performance. 