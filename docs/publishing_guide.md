# Publishing Quantized Models to Hugging Face Hub

This comprehensive guide provides detailed instructions for publishing quantized models on the Hugging Face Hub, with specific considerations for model optimization while respecting intellectual property rights. It's designed to help you share your quantized models effectively with the community.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Publishing Methods](#publishing-methods)
4. [Preparing Your Model for Publication](#preparing-your-model-for-publication)
5. [Creating an Effective Model Card](#creating-an-effective-model-card)
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

If you prefer more control over the publishing process, you can manually publish your model using the steps outlined in the [Technical Implementation](#technical-implementation) section.

## Preparing Your Model for Publication

### Benchmark and Evaluate Before Publishing

Before publishing, thoroughly evaluate your quantized model to ensure it meets quality standards:

```bash
# Run the benchmark script on your quantized model
python benchmark_your_model.py --original "microsoft/Phi-4-mini-instruct" --quantized "your-quantized-model-path" --device cpu
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

## Creating an Effective Model Card

A comprehensive model card is essential for users to understand your quantized model. Create a `README.md` file with the following sections:

### Essential Components for Quantized Models

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

### Example Model Card Structure

```markdown
---
language: en
license: [original model license]
tags:
  - quantized
  - [original model tags]
  - [quantization method, e.g., gptq, bitsandbytes, awq]
  - [bit width, e.g., 4bit, 8bit]
datasets:
  - [datasets used for evaluation]
---

# [Model Name] Quantized ([Quantization Method], [Bit Width])

This is a quantized version of the [original model name](link to original model) using [quantization method] at [bit width] precision.

## Model Description

[Brief description of the original model]

## Quantization Details

- **Original Model**: [original model name with link]
- **Quantization Method**: [method used, e.g., GPTQ, BitsAndBytes, AWQ]
- **Bit Width**: [bit width, e.g., 4-bit, 8-bit]
- **Group Size**: [group size used, if applicable]
- **Quantization Tool**: [Model Quantizer](https://github.com/lpalbou/model-quantizer)

## Performance Comparison

| Metric | Original Model | Quantized Model |
| ------ | -------------- | --------------- |
| Memory Usage | [value] GB | [value] GB |
| Loading Time | [value] s | [value] s |
| Inference Speed | [value] tokens/s | [value] tokens/s |
| [Quality Metric] | [value] | [value] |

## Hardware Requirements

- **Minimum RAM**: [value] GB
- **Recommended VRAM**: [value] GB
- **Supported Devices**: [list of tested devices]

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained("YOUR_HF_USERNAME/MODEL_NAME", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_USERNAME/MODEL_NAME")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

[Discuss any limitations or quality degradation observed with the quantized model]

## Citation

[Include citation for the original model]
```

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
python benchmark_your_model.py --original microsoft/Phi-4-mini-instruct --quantized phi-4-mini-quantized --device cpu
```

Document the results for inclusion in your model card.

### Step 4: Create a Comprehensive Model Card

Create a `README.md` file with detailed information about your quantized model:

```markdown
---
language: en
license: microsoft-research-license
tags:
  - quantized
  - gptq
  - 4bit
  - phi-4
  - microsoft
---

# Phi-4-mini Quantized (GPTQ 4-bit)

This repository contains a 4-bit quantized version of [Microsoft's Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct) optimized for efficient inference.

## Model Description

Phi-4-mini is a compact yet powerful language model released by Microsoft Research. This repository provides an optimized version using GPTQ quantization to reduce memory requirements while maintaining performance.

## Quantization Details

- **Original Model**: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- **Quantization Method**: GPTQ
- **Bit Width**: 4-bit
- **Group Size**: 128
- **Quantization Tool**: [Model Quantizer](https://github.com/lpalbou/model-quantizer)

## Performance Comparison

| Metric | Original | 4-bit Quantized |
|--------|----------|----------------|
| Memory Usage | ~7.6 GB | ~1.9 GB |
| Loading Time | X.XX s | Y.YY s |
| Generation Speed | XX.X tokens/s | YY.Y tokens/s |

## Hardware Requirements

- **Minimum RAM**: 4 GB
- **Recommended VRAM**: 2 GB
- **Supported Devices**: CPU, CUDA GPUs, MPS (Apple Silicon)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    "your-username/phi-4-mini-gptq-4bit",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("your-username/phi-4-mini-gptq-4bit")

# Generate text
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Format Clarification

This quantized model uses the Hugging Face GPTQ format, which is different from other quantization formats like GGUF used by llama.cpp. This model is designed to be used with the Hugging Face Transformers library.

## Limitations

- The 4-bit quantization may result in slight quality degradation for complex reasoning tasks
- Performance may vary depending on hardware and specific use cases
- [Any other limitations observed during testing]

## License

This model is subject to the [Microsoft Research License](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/LICENSE). This is a quantized version of the original model created by Microsoft.

## Citation

If you use this model, please cite both the original Microsoft Phi-4-mini model and this quantized version.
```

### Step 5: Publish to Hugging Face

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
    commit_message="Upload 4-bit quantized Phi-4-mini model"
)
```

### Step 6: Verify and Engage

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