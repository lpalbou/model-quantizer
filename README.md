# Model Quantizer

A tool for quantizing and saving Hugging Face models, with comprehensive benchmarking and testing capabilities.

[![PyPI version](https://badge.fury.io/py/model-quantizer.svg)](https://badge.fury.io/py/model-quantizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why Model Quantizer?

- **Cross-Platform Compatibility**: BitsAndBytes doesn't work on macOS/OSX, but Hugging Face GPTQ implementation does
- **Pre-Quantized Models**: Quantizing models with GPTQ takes time, so we provide tools to publish pre-quantized models for reuse
- **Control Over Quantization**: Unlike other published quantized models, this tool gives you full control over the quantization process
- **Test Before Publishing**: Comprehensive benchmarking and testing tools to validate your quantized model's performance before publishing
- **Easy Publishing**: Streamlined process to publish quantized models to Hugging Face Hub

## Features

- Integrates with Hugging Face's quantization methods:
  - GPTQ: Post-training quantization using Hugging Face's implementation of the GPTQ algorithm (primarily for 4-bit and 8-bit)
  - BitsAndBytes: Quantization using the BitsAndBytes library
  - AWQ: Post-training quantization using the AWQ algorithm
- Supports different bit widths based on the method:
  - GPTQ: Primarily designed for 4-bit and 8-bit quantization
  - BitsAndBytes: 4-bit and 8-bit quantization
  - AWQ: 4-bit quantization
- Saves quantized models for later use
- Command-line interface for easy use
- Python API for integration into other projects
- Comprehensive benchmarking tools to compare original and quantized models
- Interactive testing capabilities to verify model quality

## Core Workflow

The Model Quantizer provides a complete workflow for working with quantized models:

1. **Quantize**: Convert your model to a more efficient format
2. **Benchmark**: Compare performance metrics between original and quantized versions
3. **Test**: Interact with your model to verify quality and responsiveness
4. **Publish**: Share your optimized model with the community

## Installation

### From PyPI

```bash
pip install model-quantizer
```

### From Source

```bash
git clone https://github.com/lpalbou/model-quantizer.git
cd model-quantizer
pip install -e .
```

## Usage

### Command-Line Interface

```bash
# Basic usage with GPTQ (4-bit recommended)
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method gptq

# Specify output directory
model-quantizer microsoft/Phi-4-mini-instruct --output-dir phi-4-mini-quantized

# Use BitsAndBytes quantization
model-quantizer microsoft/Phi-4-mini-instruct --bits 4 --method bitsandbytes

# Specify device
model-quantizer microsoft/Phi-4-mini-instruct --device mps

# Custom calibration dataset
model-quantizer microsoft/Phi-4-mini-instruct --calibration-dataset "This is a sample text,This is another sample"
```

### Python API

```python
from quantizer import ModelQuantizer, QuantizationConfig

# Create configuration
config = QuantizationConfig(
    bits=4,  # 4-bit recommended for GPTQ
    method="gptq",
    output_dir="quantized-model",
    device="auto"
)

# Create quantizer
quantizer = ModelQuantizer(config)

# Quantize model
model, tokenizer = quantizer.quantize("microsoft/Phi-4-mini-instruct")

# Save model
quantizer.save()

# Load quantized model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("quantized-model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("quantized-model")
```

## Configuration Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `bits` | Number of bits for quantization (4 or 8 recommended for GPTQ) | 8 |
| `method` | Quantization method to use (gptq, awq, or bitsandbytes) | gptq |
| `output_dir` | Directory to save the quantized model | quantized-model |
| `calibration_dataset` | Dataset for calibration | Default dataset |
| `group_size` | Group size for quantization | 128 |
| `desc_act` | Whether to use descending activation order | False |
| `sym` | Whether to use symmetric quantization | True |
| `device` | Device to use for quantization (auto, cpu, cuda, mps) | auto |
| `use_optimum` | Whether to use the optimum library for quantization | True |

## Benchmarking and Testing Tools

The package includes several tools to help you benchmark and test your quantized models:

### Benchmarking Models

Benchmarking is a crucial step before publishing your quantized model. It helps you verify that the quantization process maintained acceptable performance while reducing memory usage.

#### Choosing Between Benchmarking Tools

The Model Quantizer provides two main benchmarking tools:

##### run-benchmark

`run-benchmark` is an all-in-one solution that:
- Runs the benchmark comparing original and quantized models
- Saves the results to a JSON file
- Generates a visual HTML report
- Opens the report (on macOS) or provides the path to the report

Use `run-benchmark` when:
- You want a complete end-to-end benchmarking solution
- You need visual reports generated automatically
- You're doing a one-off comparison between models
- You want a simple, streamlined process

```bash
# Quick comparison with visual report
run-benchmark --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_tokens 50 --output_dir benchmark_results
```

##### benchmark-model

`benchmark-model` is a more flexible, lower-level tool that:
- Runs benchmarks with more customizable parameters
- Outputs raw benchmark data
- Can be integrated into custom workflows

Use `benchmark-model` when:
- You need more granular control over benchmark parameters
- You want to run multiple benchmarks and analyze the results yourself
- You're integrating benchmarking into a custom workflow or script
- You want to compare multiple models or configurations systematically

```bash
# Fine-tuning benchmark parameters
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_new_tokens 100 --temperature 0.7 --top_p 0.9 --output custom_benchmark.json

# Batch testing with different prompt sets
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized ./quantized-model --prompts-file scientific_prompts.json --output scientific_benchmark.json
```

#### Using the Automated Benchmark Script (Recommended)

The `run-benchmark` script provides a convenient one-step process for benchmarking and visualization:

```bash
# Run the complete benchmark process with required parameters
run-benchmark --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --device cpu --max_tokens 50 --output_dir benchmark_results
```

**Available Options:**
- `--original MODEL_PATH`: Path to the original model (required)
- `--quantized MODEL_PATH`: Path to the quantized model (required)
- `--device DEVICE`: Device to use for benchmarking (cpu, cuda, mps) (required)
- `--max_tokens NUM`: Maximum number of tokens to generate (required)
- `--output_dir DIR`: Directory to save benchmark results (required)
- `--quiet`: Run in quiet mode with minimal output

The script automatically:
1. Runs the benchmark comparing the original and quantized models
2. Saves the results to a JSON file
3. Generates a visual HTML report
4. Opens the report (on macOS) or provides the path to the report

#### Using the Core Benchmarking Tool Directly

For more control over the benchmarking process, you can use `benchmark-model` directly:

```bash
# Basic usage - compare original and quantized models
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --device cpu

# Specify maximum tokens to generate
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --max_new_tokens 100

# Save benchmark results to a file
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --output benchmark_results.json

# Run with reduced output verbosity
benchmark-model --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --quiet

# Compare two different quantized models
benchmark-model --original qmodels/phi4-mini-8bit --quantized qmodels/phi4-mini-4bit --device cpu
```

**Key Parameters:**
- `--original`: Path to the original or baseline model
- `--quantized`: Path to the quantized or comparison model
- `--device`: Device to run models on (cpu, cuda, mps)
- `--max_new_tokens`: Maximum number of tokens to generate (default: 100)
- `--output`: Path to save benchmark results as JSON
- `--quiet`: Reduce output verbosity
- `--num_prompts`: Number of prompts to use per category (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)

### Visualizing Benchmark Results

After running the benchmark, you can generate visual reports from the results:

```bash
# Generate visual report from benchmark results
visualize-benchmark --input benchmark_results.json --output_dir benchmark_report

# Open the HTML report
open benchmark_report/benchmark_report.html
```

The visualization script generates:
- An HTML report with detailed metrics
- Charts comparing memory usage
- Charts comparing performance metrics
- Charts comparing performance by prompt category

### Interactive Testing with Chat

Before publishing, it's important to test your model interactively to ensure it maintains response quality after quantization:

```bash
# Chat with the model
chat-with-model --model_path qmodels/phi4-mini-4bit --device cpu --max_new_tokens 256

# Use a custom system prompt
chat-with-model --model_path qmodels/phi4-mini-4bit --system_prompt "You are a helpful AI assistant specialized in science."
```

## Benchmark Metrics

The `benchmark-model` script provides comprehensive metrics comparing original and quantized models:

### Memory Usage
- Initial memory: Memory usage before loading the model
- Min memory: Minimum memory usage during the benchmark
- Max memory: Maximum memory usage during the benchmark
- Avg memory: Average memory usage during the benchmark
- Memory increase: Additional memory used after loading the model

### Loading Performance
- Load time: Time taken to load the model

### Generation Performance
- Prompt tokens: Number of tokens in the input prompts
- Prompt eval time: Time spent processing input prompts
- Prompt tokens/sec: Rate of processing input tokens
- Generated tokens: Number of tokens generated
- Generation time: Time spent generating tokens
- Generation tokens/sec: Rate of generating output tokens

### Prompt Categories
The benchmark tests different types of prompts:
- Short prompts (< 100 tokens)
- Medium prompts (100-500 tokens)
- Long prompts (500-1000 tokens)
- Very long prompts (1000+ tokens)

Each category includes factual, creative, and reasoning tasks to test different model capabilities.

## Documentation

Comprehensive documentation is available in the [docs](docs) directory:

- [General Model Quantization Guide](docs/general_guide.md): A guide to quantizing any Hugging Face model
- [Phi-4-Mini Quantization Guide](docs/phi4_mini.md): Specific guide for quantizing the Phi-4-mini model
- [Benchmarking Guide](docs/benchmarking.md): How to benchmark quantized models
- [Troubleshooting Guide](docs/troubleshooting.md): Solutions for common issues
- [Publishing Guide](docs/publishing_guide.md): How to publish quantized models to Hugging Face Hub

## Examples

See the [examples](examples) directory for more examples.

### Example: Quantizing Any Model

The `examples/quantize_example.py` script demonstrates how to quantize any Hugging Face model:

```bash
# Quantize a model with default settings (Phi-4-mini with GPTQ 4-bit)
python examples/quantize_example.py

# Quantize a specific model
python examples/quantize_example.py --model facebook/opt-350m --method gptq --bits 4

# Publish to Hugging Face Hub
python examples/quantize_example.py --model facebook/opt-350m --publish --repo-id your-username/opt-350m-gptq-4bit
```

The script provides a complete workflow for quantizing, benchmarking, testing, and publishing models.

### Example: Quantizing Phi-4-mini

The Phi-4-mini model from Microsoft is a great example of a model that benefits from quantization. At 3.8B parameters, it can be quantized to reduce memory usage:

- Original model (FP16): Theoretical ~7.6 GB memory usage
- 8-bit quantized: Theoretical ~3.8 GB memory usage (50% reduction)
- 4-bit quantized: Theoretical ~1.9 GB memory usage (75% reduction)

**Note**: These are theoretical estimates based on bit reduction. Actual memory usage may vary and should be benchmarked for your specific hardware and use case. The quantized models produced by this tool use the Hugging Face GPTQ format, which is different from other formats like GGUF.

Our benchmarks with the Phi-4-mini model show:

1. **8-bit Quantized Model**:
   - Theoretical memory reduction: 50% of original model
   - Loading time: ~0.91 seconds
   - Generation time: Slower than 4-bit model

2. **4-bit Quantized Model**:
   - Theoretical memory reduction: 75% of original model
   - Loading time: ~1.50 seconds
   - Generation time: Faster than 8-bit model

```bash
# Quantize using GPTQ with 4-bit precision (recommended)
python examples/quantize_example.py --model microsoft/Phi-4-mini-instruct --method gptq --bits 4 --output-dir ./quantized-models/phi4-mini-gptq-4bit

# Quantize using BitsAndBytes with 8-bit precision
python examples/quantize_example.py --model microsoft/Phi-4-mini-instruct --method bnb --bits 8 --output-dir ./quantized-models/phi4-mini-bnb-8bit
```

See the [Phi-4-Mini Quantization Guide](docs/phi4_mini.md) for more details.

## Notes

- The 4-bit model generally provides the best balance between memory usage and performance.
- On CPU, generation is quite slow (can take several minutes for longer responses).
- For optimal performance, use a GPU if available.
- The quantized models maintain good response quality compared to the original model.
- Always benchmark and test your model before publishing to ensure it meets your quality standards.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Optimum 1.12+ (for GPTQ quantization)
- BitsAndBytes 0.40+ (for BitsAndBytes quantization)
- AutoAWQ 0.1+ (for AWQ quantization)
- psutil (for memory tracking)
- numpy (for statistical calculations)
- matplotlib (for visualization)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Publishing Models

After quantizing and benchmarking your model, you can publish it to the Hugging Face Hub:

```bash
model-quantizer microsoft/phi-4-mini-instruct --bits 4 --method gptq --publish --repo-id YOUR_USERNAME/phi-4-mini-gptq-4bit
```

This will:
1. Quantize the model
2. Save it locally
3. Generate a comprehensive model card with model details, quantization parameters, and estimated memory usage
4. Upload it to the Hugging Face Hub under the specified repository ID

You can also enhance your model card with benchmark results by running:

```bash
run-benchmark --original microsoft/phi-4-mini-instruct --quantized ./quantized-model --device cpu --max_tokens 100 --output_dir ./benchmark_results --update-model-card
```

This will automatically update the model card with memory usage, loading time, generation speed, and comparison with the original model.

For more details, see the [Publishing Guide](docs/publishing.md).