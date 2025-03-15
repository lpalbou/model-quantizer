# Publishing Quantized Models

This guide explains how to publish your quantized models to the Hugging Face Hub.

## Prerequisites

1. A Hugging Face account
2. The Hugging Face Hub CLI installed and configured
3. A quantized model that you want to publish

## Publishing During Quantization

The simplest way to publish a quantized model is to use the `--publish` flag when running the quantization command:

```bash
model-quantizer microsoft/phi-4-mini-instruct --bits 4 --method gptq --publish --repo-id YOUR_USERNAME/phi-4-mini-gptq-4bit
```

This will:
1. Quantize the model
2. Save it locally
3. Upload it to the Hugging Face Hub under the specified repository ID

## Publishing an Existing Quantized Model

If you have already quantized a model and want to publish it, you can use the Hugging Face Hub CLI:

```bash
huggingface-cli upload YOUR_USERNAME/phi-4-mini-gptq-4bit ./quantized-model
```

## Model Cards

### Automatic Model Card Generation

When you quantize a model, a comprehensive model card is automatically generated and saved as `README.md` in the output directory. This model card includes:

- Basic model information (original model, quantization method, bit width)
- Quantization parameters (group size, symmetric quantization, etc.)
- Estimated memory usage compared to the original model
- Usage examples
- License information

### Enhancing Model Cards with Benchmark Results

To add benchmark results to your model card, you can run the benchmark with the `--update-model-card` flag:

```bash
benchmark-model --original microsoft/phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_new_tokens 100 --output ./benchmark_results.json --update-model-card
```

Or when using the all-in-one benchmark tool:

```bash
run-benchmark --original microsoft/phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_tokens 100 --output_dir ./benchmark_results --update-model-card
```

This will automatically update the model card with:

- Memory usage metrics
- Loading time
- Generation speed
- Comparison with the original model
- Quality metrics (if available)

### Model Card Example

Here's an example of what the generated model card looks like:

```markdown
---
language:
- en
tags:
- quantized
- GPTQ
- 4bit
license: mit
datasets:
- c4
---

# phi-4-mini-gptq-4bit

This is a 4-bit quantized version of [microsoft/phi-4-mini-instruct](https://huggingface.co/microsoft/phi-4-mini-instruct) using the gptq quantization method.

## Model Details

- **Original Model:** [microsoft/phi-4-mini-instruct](https://huggingface.co/microsoft/phi-4-mini-instruct)
- **Quantization Method:** gptq (4-bit)
- **Hugging Face Transformers Compatible:** Yes
- **Quantized Date:** 2023-06-15
- **Quantization Parameters:**
  - Group Size: 128
  - Bits: 4
  - Descending Activation Order: False
  - Symmetric: True

## Performance Metrics

### Benchmark Results

#### Memory Metrics
| Metric | Value |
| ------ | ----- |
| Initial Memory | 0.45 GB |
| Min Memory | 0.45 GB |
| Max Memory | 2.10 GB |
| Avg Memory | 1.85 GB |

#### Performance Metrics
| Metric | Value |
| ------ | ----- |
| Load Time | 3.25 s |
| Prompt Tokens Per Sec | 45.32 tokens/s |
| Generation Tokens Per Sec | 12.75 tokens/s |

#### Model Comparison
| Model | Memory | Load Time | Generation Speed | Quality |
| ----- | ------ | --------- | ---------------- | ------- |
| Original | 4.50 GB | 5.32 s | 10.45 tokens/s | Baseline |
| Quantized | 2.10 GB | 3.25 s | 12.75 tokens/s | See metrics |

### Memory Usage
- Original Model (FP16): ~8.4 GB
- Quantized Model (4-bit): ~2.1 GB
- Memory Reduction: ~75.0%

### Speed
- Load Time: 3.25 s
- Generation Speed: 12.75 tokens/sec

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/phi-4-mini-gptq-4bit", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/phi-4-mini-gptq-4bit")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Quantization Process

This model was quantized using the [Model Quantizer](https://github.com/lpalbou/model-quantizer) tool with the following command:

```bash
model-quantizer microsoft/phi-4-mini-instruct --bits 4 --method gptq --output-dir ./quantized-model
```

## License

This model is licensed under the same license as the original model: mit.
```

## Best Practices for Publishing

1. **Run benchmarks before publishing**: Always benchmark your quantized model against the original to understand the performance trade-offs.

2. **Include benchmark results in your model card**: This helps users understand what to expect from your quantized model.

3. **Choose a descriptive repository name**: Include the model name, quantization method, and bit width in the repository name (e.g., `phi-4-mini-gptq-4bit`).

4. **Set appropriate tags**: Include tags like `quantized`, `gptq`, `4bit`, etc. to make your model discoverable.

5. **Test your model before publishing**: Make sure your model works as expected by testing it with the Hugging Face Transformers library.

## Troubleshooting

- **Upload fails**: Make sure you have the correct permissions to upload to the specified repository.
- **Model doesn't work after upload**: Ensure all necessary files are included in the upload (model files, tokenizer files, configuration files).
- **Model card not showing**: Check that the README.md file was properly uploaded to the repository. 