# Publishing Quantized Models to Hugging Face Hub

This comprehensive guide provides detailed instructions for publishing quantized models on the Hugging Face Hub, with specific considerations for model optimization while respecting intellectual property rights. It's designed to help you share your quantized models effectively with the community.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Publishing Methods](#publishing-methods)
4. [Preparing Your Model for Publication](#preparing-your-model-for-publication)
5. [Model Cards](#model-cards)
6. [Legal and Ethical Considerations](#legal-and-ethical-considerations)
7. [Best Practices for Model Organization](#best-practices-for-model-organization)
8. [Community Engagement and Maintenance](#community-engagement-and-maintenance)
9. [Technical Implementation](#technical-implementation)
10. [Example: Publishing Quantized Versions of Phi-4-mini](#example-publishing-quantized-versions-of-phi-4-mini)

## Introduction

Publishing quantized models on Hugging Face makes optimized AI accessible to users with limited computational resources. This guide focuses on best practices for sharing models that have been quantized using methods such as:

- **GPTQ**: A state-of-the-art one-shot weight quantization method
- **BitsAndBytes**: Quantization using the BitsAndBytes library
- **AWQ**: Activation-aware weight quantization for efficient inference
- **Other PTQ methods**: Including various post-training quantization approaches

These techniques produce models in various precisions:

- **4-bit Models**: Maximum memory efficiency with reasonable quality preservation
- **8-bit Models**: Balanced approach for wider compatibility

## Prerequisites

Before publishing your quantized model, ensure you have:

1. Successfully quantized a model using the Model Quantizer tool
2. Thoroughly benchmarked and tested your quantized model
3. A Hugging Face account (sign up at [huggingface.co](https://huggingface.co/join))
4. The Hugging Face CLI installed (`pip install huggingface_hub`)
5. Logged in to Hugging Face CLI (`huggingface-cli login`)

## Publishing Methods

Model Quantizer provides three ways to publish your quantized models to Hugging Face Hub:

### 1. Using the Command-Line Interface (Recommended)

The simplest way to publish your model is to use the `--publish` flag when quantizing:

```bash
# Quantize and publish in one step
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq --publish --repo-id "your-username/phi-4-mini-gptq-4bit"
```

Additional publishing options:

```bash
# Make the repository private
model-quantizer microsoft/Phi-4-mini-instruct --publish --private

# Specify a custom commit message
model-quantizer microsoft/Phi-4-mini-instruct --publish --commit-message "Add quantized Phi-4-mini model"
```

### 2. Using the Python API

If you're using the Python API, you can publish your model after quantizing:

```python
from quantizer import ModelQuantizer, QuantizationConfig

# Create configuration
config = QuantizationConfig(
    bits=4,
    method="gptq",
    output_dir="phi-4-mini-quantized"
)

# Create quantizer
quantizer = ModelQuantizer(config)

# Quantize model
model, tokenizer = quantizer.quantize("microsoft/Phi-4-mini-instruct")

# Publish to Hugging Face Hub
quantizer.publish_to_hub(
    repo_id="your-username/phi-4-mini-gptq-4bit",
    private=False,
    commit_message="Upload quantized Phi-4-mini model"
)
```

### 3. Manual Publishing

If you have already quantized a model and want to publish it manually, you can use the Hugging Face Hub CLI:

```bash
huggingface-cli upload YOUR_USERNAME/phi-4-mini-gptq-4bit ./quantized-model
```

For more control over the publishing process, you can follow the steps outlined in the [Technical Implementation](#technical-implementation) section.

## Preparing Your Model for Publication

### Benchmark and Evaluate Before Publishing

Before publishing, thoroughly evaluate your quantized model to ensure it meets quality standards:

```bash
# Run the benchmark script on your quantized model
run-benchmark --original "microsoft/Phi-4-mini-instruct" --quantized "your-quantized-model-path" --device cpu --output_dir benchmark_results
```

Important metrics to document:
- Performance comparison with the original model
- Memory usage reduction
- Inference speed improvements
- Quality metrics (perplexity, accuracy, task-specific metrics)
- Hardware compatibility

### Model Files and Organization

Ensure your quantized model includes all necessary files:
- Model weights (usually in safetensors format)
- Configuration files (config.json)
- Tokenizer files
- README.md (model card)
- LICENSE file (matching the original model's license)

Test that your model can be loaded directly via the Hugging Face API before publishing:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test loading your local model
model = AutoModelForCausalLM.from_pretrained("path/to/your/quantized/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/your/quantized/model")

# Test a simple inference
inputs = tokenizer("Test prompt", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Cards

### Automatic Model Card Generation

When you quantize a model using Model Quantizer, a comprehensive model card is automatically generated and saved as `README.md` in the output directory. This model card includes:

- Basic model information (original model, quantization method, bit width)
- Quantization parameters (group size, symmetric quantization, etc.)
- Estimated memory usage compared to the original model
- Usage examples
- License information

### Enhancing Model Cards with Benchmark Results

To add benchmark results to your model card, you can run the benchmark with the `--update-model-card` flag:

```bash
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_new_tokens 100 --output ./benchmark_results.json --update-model-card
```

Or when using the all-in-one benchmark tool:

```bash
run-benchmark --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_tokens 100 --output_dir ./benchmark_results --update-model-card
```

This will automatically update the model card with:

- Memory usage metrics
- Loading time
- Generation speed
- Comparison with the original model
- Quality metrics (if available)

### Model Card Example

Here's an example of what the automatically generated model card looks like:

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

### Creating a Custom Model Card

If you prefer to create a custom model card, you can follow the guidelines below to ensure it includes all necessary information.

#### Essential Components for Quantized Models

- **Model Description**: Clearly state it's a quantized version of an existing model
- **Quantization Details**: 
  - Method used (GPTQ, BitsAndBytes, AWQ)
  - Bit precision (4-bit, 8-bit)
  - Group size and other parameters used
  - Quantization tool used (Model Quantizer)
- **Performance Comparison**:
  - Benchmark results against the original model
  - Memory usage reduction
  - Inference speed improvements
  - Quality metrics
- **Hardware Requirements**:
  - Minimum system specifications
  - VRAM/RAM requirements
  - Supported accelerators
- **Usage Examples**:
  - Loading with quantization-aware code
  - Inference examples
- **Limitations**:
  - Any known limitations or quality degradation
  - Specific use cases where performance might be affected
- **Citation Information**:
  - How to cite both your quantized model and the original model

## Legal and Ethical Considerations

### License Compliance

- Verify the original model's license permits redistribution and modification
- Common permissive licenses: MIT, Apache 2.0, BSD
- Research-specific licenses may have additional restrictions
- Ensure your quantized model includes the same license as the original model

### Attribution Requirements

- Always credit the original model creators
- Link to the original model repository
- Maintain the same license as the original model
- Clearly differentiate your contribution (quantization) from the original work

### Ethical Distribution

- Document any performance degradation or biases introduced by quantization
- Specify intended use cases and limitations
- Include warnings about potential issues in critical applications
- Be aware that quantizing a model does not change the intellectual property rights of the original model weights

## Best Practices for Model Organization

### Naming Conventions

Follow these naming conventions for your quantized models:

- **Repository name**: `[original-model-name]-[quantization-method]-[bit-width]`
  - Example: `phi-4-mini-gptq-4bit`

### Tagging

Add appropriate tags to your model to make it discoverable:

- `quantized`: Indicates this is a quantized model
- `[quantization method]`: The method used (e.g., `gptq`, `bitsandbytes`, `awq`)
- `[bit width]`: The precision used (e.g., `4bit`, `8bit`)
- Original model tags: Include relevant tags from the original model

### Format Clarification

Make sure to clarify in your model card that your quantized model uses the Hugging Face format (GPTQ, BitsAndBytes, or AWQ), which is different from other quantization formats like GGUF used by llama.cpp. This helps users understand the compatibility of your model with different frameworks.

### Model Card Organization

Organize your model card with clear sections:

1. **Introduction**: Brief overview of the quantized model
2. **Quantization Details**: Technical details about the quantization process
3. **Performance Comparison**: Metrics comparing original and quantized models
4. **Usage Examples**: Code snippets showing how to use the model
5. **Limitations**: Any known limitations or issues
6. **Citation**: Proper attribution to the original model

## Community Engagement and Maintenance

### Supporting Users

- Monitor discussions and issues on your model repository
- Provide guidance on hardware requirements and compatibility
- Help troubleshoot loading and inference problems
- Respond to user questions in a timely manner

### Collecting Feedback

- Encourage users to share their benchmark results on different hardware
- Document community findings in model card updates
- Consider community contributions for improvements
- Create a discussion thread for users to share their experiences

### Long-term Maintenance

- Update with newer quantization techniques as they become available
- Rebase on new versions of the original model when available
- Document compatibility with evolving libraries
- Keep the model card updated with the latest information

## Technical Implementation

### Step-by-Step Publishing Process

1. **Install Required Libraries**
   ```bash
   pip install transformers huggingface_hub
   ```

2. **Login to Hugging Face**
   ```bash
   huggingface-cli login
   ```

3. **Create a New Repository**
   ```bash
   huggingface-cli repo create MODEL_NAME-[quantization-method]-[bit-width]
   ```

4. **Upload Your Model**

   Using the `transformers` library:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Load your quantized model
   model = AutoModelForCausalLM.from_pretrained("PATH_TO_YOUR_QUANTIZED_MODEL")
   tokenizer = AutoTokenizer.from_pretrained("PATH_TO_YOUR_QUANTIZED_MODEL")

   # Push to Hub
   model.push_to_hub("YOUR_HF_USERNAME/MODEL_NAME-[quantization-method]-[bit-width]")
   tokenizer.push_to_hub("YOUR_HF_USERNAME/MODEL_NAME-[quantization-method]-[bit-width]")
   ```

   Or using the Hugging Face Hub API:
   ```python
   from huggingface_hub import HfApi, HfFolder
   import os

   # Authentication with Hugging Face
   api = HfApi()
   api.set_access_token(HfFolder.get_token())

   # Define repository name for Hugging Face
   repo_name = "YOUR_HF_USERNAME/MODEL_NAME-[quantization-method]-[bit-width]"

   # Create the repository if it doesn't exist
   try:
       api.create_repo(repo_name)
       print(f"Created new repository: {repo_name}")
   except Exception as e:
       print(f"Repository already exists or error: {e}")

   # Upload to Hugging Face
   api.upload_folder(
       folder_path="PATH_TO_YOUR_QUANTIZED_MODEL",
       path_in_repo=".",
       repo_id=repo_name,
       commit_message="Upload quantized model with documentation"
   )
   ```

   Alternatively, using the command line:
   ```bash
   # Navigate to your quantized model directory
   cd PATH_TO_YOUR_QUANTIZED_MODEL

   # Upload to Hugging Face Hub
   huggingface-cli upload YOUR_HF_USERNAME/MODEL_NAME-[quantization-method]-[bit-width] .
   ```

5. **Verify Your Upload**

   Visit your model page on the Hugging Face Hub to verify that all files were uploaded correctly:
   ```
   https://huggingface.co/YOUR_HF_USERNAME/MODEL_NAME-[quantization-method]-[bit-width]
   ```

## Example: Publishing Quantized Versions of Phi-4-mini

Here's an end-to-end example of preparing and publishing a quantized version of Microsoft's Phi-4-mini model to Hugging Face.

### Step 1: Verify License Compatibility

Before publishing, verify the license terms of the original model:

Microsoft's Phi-4-mini-instruct uses the Microsoft Research License, which allows for:
- Research use
- Adaptation (including quantization)
- Redistribution with proper attribution

**Key License Requirements:**
- Maintain the original license in your repository
- Clearly attribute Microsoft as the original creator
- Specify that you've created a quantized version, not an original model

### Step 2: Quantize the Model

Use the Model Quantizer to create a 4-bit quantized version:

```bash
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq --output-dir phi-4-mini-quantized
```

### Step 3: Benchmark the Quantized Model

Compare the performance of the original and quantized models:

```bash
run-benchmark --original microsoft/Phi-4-mini-instruct --quantized phi-4-mini-quantized --device cpu --output_dir benchmark_results --update-model-card
```

The benchmark results will be automatically added to your model card.

### Step 4: Publish to Hugging Face

```bash
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq --publish --repo-id "your-username/phi-4-mini-gptq-4bit"
```

### Step 5: Verify and Engage

After publishing:
1. Verify all files were uploaded correctly
2. Test loading the model from Hugging Face
3. Monitor for user feedback and questions
4. Update the model card with additional information as needed

## References

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)
- [Hugging Face CLI Documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face GPTQ Documentation](https://huggingface.co/docs/transformers/en/quantization/gptq)
- [Model Quantizer Documentation](https://github.com/lpalbou/model-quantizer)
- [Model Quantizer PyPI Package](https://pypi.org/project/model-quantizer/) 