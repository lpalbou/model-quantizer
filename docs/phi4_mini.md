# Phi-4-Mini Quantization Guide

This guide explains how to quantize the Microsoft Phi-4-mini-instruct model using the Model Quantizer tool.

## About Phi-4-Mini

Phi-4-mini-instruct is a lightweight open model built upon synthetic data and filtered publicly available websites - with a focus on high-quality, reasoning dense data. The model belongs to the Phi-4 model family and supports 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning and direct preference optimization to support precise instruction adherence and robust safety measures.

- Model size: 3.8B parameters
- Context length: 128K tokens
- Architecture: Dense decoder-only Transformer model

## Why Quantize Phi-4-Mini?

Phi-4-mini is an excellent candidate for quantization for several reasons:

1. **Memory Efficiency**: Even at 3.8B parameters, the model requires significant memory in its original form (~7.6 GB in FP16). Quantization can reduce this to ~3.8 GB (8-bit) or ~1.9 GB (4-bit), making it accessible on devices with limited memory.

2. **Cross-Platform Compatibility**: While BitsAndBytes quantization doesn't work on macOS, GPTQ quantization does. This is particularly important for Phi-4-mini, which is designed to be accessible to a wide range of users, including those on macOS.

3. **Minimal Quality Loss**: Phi-4-mini maintains most of its capabilities even when quantized to 4 bits, making it an ideal candidate for aggressive quantization.

4. **Inference Speed**: Quantized models often have faster inference times due to reduced memory bandwidth requirements, which is particularly beneficial for Phi-4-mini's intended use cases.

## Quantization Options

The Phi-4-mini model can be quantized using different methods and bit widths:

### GPTQ Quantization

GPTQ is a post-training quantization technique where each row of the weight matrix is quantized independently to find a version of the weights that minimizes error. These weights are quantized to int4 or int8, stored as int32, and dequantized to fp16 on the fly during inference.

```bash
# 8-bit GPTQ quantization
model-quantizer microsoft/Phi-4-mini-instruct --bits 8 --method gptq --output-dir phi-4-mini-gptq-8bit

# 4-bit GPTQ quantization (recommended)
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq --output-dir phi-4-mini-gptq-4bit
```

### BitsAndBytes Quantization

BitsAndBytes is a library for quantizing models to 4 or 8 bits. It's particularly useful for models that will be run on CUDA devices.

```bash
# 8-bit BitsAndBytes quantization
model-quantizer microsoft/Phi-4-mini-instruct --bits 8 --method bitsandbytes --output-dir phi-4-mini-bnb-8bit

# 4-bit BitsAndBytes quantization
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method bitsandbytes --output-dir phi-4-mini-bnb-4bit
```

## Memory Usage Comparison

| Quantization Method | Bits | Theoretical Memory Usage | Actual Memory Usage |
| ------------------- | ---- | ------------------------ | ------------------- |
| None (FP16)         | 16   | ~7.6 GB                  | To be benchmarked   |
| GPTQ                | 8    | ~3.8 GB                  | To be benchmarked   |
| GPTQ                | 4    | ~1.9 GB                  | To be benchmarked   |
| BitsAndBytes        | 8    | ~3.8 GB                  | To be benchmarked   |
| BitsAndBytes        | 4    | ~1.9 GB                  | To be benchmarked   |

**Note**: The theoretical memory usage is calculated based on the bit reduction (50% for 8-bit, 75% for 4-bit) from the original FP16 model. Actual memory usage may vary based on hardware, implementation details, and other factors. We recommend benchmarking on your specific setup.

## Format Comparison

It's important to note that this tool produces models in the Hugging Face GPTQ format, which is different from other quantization formats like GGUF (used in repositories such as [bartowski/microsoft_Phi-4-mini-instruct-GGUF](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF)). These formats have different characteristics:

- **GPTQ (this tool)**: Integrates with Hugging Face's transformers library, making it easy to use with existing Hugging Face workflows.
- **GGUF**: Used by llama.cpp and related projects, optimized for CPU inference and different deployment scenarios.

Choose the format that best fits your deployment needs.

## Performance Considerations

- **GPTQ** generally provides better performance on CPU and Apple Silicon (MPS) devices.
- **BitsAndBytes** is optimized for CUDA devices and may not work well on CPU or MPS.
- Lower bit widths (4-bit) provide better memory efficiency but may have slightly lower quality.
- 8-bit quantization is a good balance between memory efficiency and quality.

## Loading Quantized Models

Once the model is quantized, it can be loaded using the Hugging Face Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained("phi-4-mini-gptq-4bit", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("phi-4-mini-gptq-4bit")

# Generate text
inputs = tokenizer("Human: What is the capital of France?\n\nAssistant:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

### GPTQ on macOS

If you encounter issues with GPTQ quantization on macOS, try setting the following environment variable:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This will allow PyTorch to fall back to CPU for operations not supported on MPS.

### BitsAndBytes on macOS

BitsAndBytes quantization may not work well on macOS as it's primarily designed for CUDA devices. If you're using macOS, we recommend using GPTQ quantization instead.

## Using with Phi4MiniServer

The quantized model can be used with the Phi4MiniServer to reduce memory usage while maintaining performance. The `examples` directory includes a script to update the Phi4MiniServer to use the quantized model.

### Updating the Server

```bash
python update_phi4_mini_server.py /path/to/phi4_mini_server.py /path/to/quantized/model
```

For example:

```bash
python update_phi4_mini_server.py ~/projects/notes/mnemosyne/athena/system/llm_server/phi4_mini_server.py ./quantized-models/phi4-mini-gptq-4bit
```

### Memory Usage Comparison

Using a quantized model with the Phi4MiniServer can significantly reduce memory usage:

| Model Version | Quantization | Theoretical Memory Usage | Actual Memory Usage |
|---------------|--------------|--------------------------|---------------------|
| Phi-4-mini    | None (FP16)  | ~7.6 GB                  | To be benchmarked   |
| Phi-4-mini    | GPTQ (8-bit) | ~3.8 GB                  | To be benchmarked   |
| Phi-4-mini    | GPTQ (4-bit) | ~1.9 GB                  | To be benchmarked   |

### Testing the Server

After updating and starting the server, you can test it using the provided test script:

```bash
./test_quantized_phi4_mini_server.sh
```

This script will:
1. Check if the server is running
2. Check the memory usage
3. Test text generation
4. Check memory usage after generation

## References

- [Microsoft Phi-4-mini-instruct on Hugging Face](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [Hugging Face GPTQ Documentation](https://huggingface.co/docs/transformers/en/quantization/gptq)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) 