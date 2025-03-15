# Model Quantizer

A tool for quantizing and saving Hugging Face models, with comprehensive benchmarking and testing capabilities.

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

## Workflow

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
    output_dir="phi-4-mini-quantized",
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
model = AutoModelForCausalLM.from_pretrained("phi-4-mini-quantized", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("phi-4-mini-quantized")
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

#### Using the Automated Benchmark Script (Recommended)

The `run_benchmark.sh` script provides a convenient one-step process for benchmarking and visualization:

```bash
# Run the complete benchmark process with required parameters
./run_benchmark.sh --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --device cpu --max_tokens 50 --output_dir benchmark_results
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

For more control over the benchmarking process, you can use `benchmark_your_model.py` directly:

```bash
# Basic usage - compare original and quantized models
python benchmark_your_model.py --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --device cpu

# Specify maximum tokens to generate
python benchmark_your_model.py --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --max_new_tokens 100

# Save benchmark results to a file
python benchmark_your_model.py --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --output benchmark_results.json

# Run with reduced output verbosity
python benchmark_your_model.py --original microsoft/Phi-4-mini-instruct --quantized qmodels/phi4-mini-4bit --quiet

# Compare two different quantized models
python benchmark_your_model.py --original qmodels/phi4-mini-8bit --quantized qmodels/phi4-mini-4bit --device cpu
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
python visualize_benchmark.py --input benchmark_results.json --output_dir benchmark_report

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
# Chat with your quantized model
python chat_with_model.py --model_path qmodels/phi4-mini-4bit --device cpu --max_new_tokens 256

# Use a custom system prompt
python chat_with_model.py --model_path qmodels/phi4-mini-4bit --system_prompt "You are a helpful AI assistant specialized in science."

# Save chat history for later review
# (Type 'save' during the chat session)
```

## Benchmark Metrics

The `benchmark_your_model.py` script provides comprehensive metrics comparing original and quantized models:

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

### Example: Quantizing Phi-4-mini

The Phi-4-mini model from Microsoft is a great example of a model that benefits from quantization. At 3.8B parameters, it can be quantized to reduce memory usage:

- Original model (FP16): Theoretical ~7.6 GB memory usage
- 8-bit quantized: Theoretical ~3.8 GB memory usage (50% reduction)
- 4-bit quantized: Theoretical ~1.9 GB memory usage (75% reduction)

**Note**: These are theoretical estimates based on bit reduction. Actual memory usage may vary and should be benchmarked for your specific hardware and use case. The quantized models produced by this tool use the Hugging Face GPTQ format, which is different from other formats like GGUF used in some repositories.

Our benchmarks with the Phi-4-mini model show:

1. **8-bit Quantized Model** (`qmodels/phi4-mini-8bit`):
   - Theoretical memory reduction: 50% of original model
   - Actual memory increase during loading: ~0.08 GB
   - Loading time: ~0.91 seconds
   - Generation time: Slower than 4-bit model

2. **4-bit Quantized Model** (`qmodels/phi4-mini-4bit`):
   - Theoretical memory reduction: 75% of original model
   - Actual memory increase during loading: ~0.24 GB
   - Loading time: ~1.50 seconds
   - Generation time: Faster than 8-bit model

```bash
# Quantize using GPTQ with 4-bit precision (recommended)
python examples/quantize_phi4_mini.py --method gptq --bits 4 --output_dir ./quantized-models/phi4-mini-gptq-4bit

# Quantize using BitsAndBytes with 8-bit precision
python examples/quantize_phi4_mini.py --method bnb --bits 8 --output_dir ./quantized-models/phi4-mini-bnb-8bit
```

See the [Phi-4-Mini Quantization Guide](docs/phi4_mini.md) for more details.

### GGUF Model Support

In addition to the Hugging Face quantization methods, we now support using pre-quantized GGUF models with our server implementation:

- **Memory Efficiency**: Our testing shows that a 6-bit quantized GGUF model (Q6_K_L) uses approximately 3.57 GB of memory when loaded for Phi-4-Mini and 4.54 GB for the full Phi-4 model
- **Server Implementation**: The server provides a simple REST API for generating text, with endpoints for health checks, text generation, and graceful shutdown
- **Current Limitations**: The server implementation is still in development and may experience stability issues during text generation with certain model configurations

```bash
# Run the Phi-4-Mini server with a 6-bit GGUF model
python -m system.llm_server.phi4_mini_server

# For the full Phi-4 model
python -m system.llm_server.phi4_server
```

When using GGUF models, ensure you have the correct model filenames configured in the server implementation:
- For Phi-4-Mini: `phi-4-mini-instruct-Q6_K_L.gguf`
- For Phi-4: `phi-4-Q6_K_L.gguf`

### Other Example Scripts

The examples directory contains several scripts to help you get started:

- `quantize_phi4_mini.py`: Script to quantize the Microsoft Phi-4-mini-instruct model
- `benchmark_your_model.py`: Comprehensive script to compare original and quantized models
- `visualize_benchmark.py`: Script to generate visual reports from benchmark results
- `run_benchmark.sh`: Shell script to automate the benchmark and visualization process
- `chat_with_model.py`: Interactive chat script for testing model quality and performance
- `update_phi4_mini_server.py`: Script to update a server to use a quantized model
- `test_quantized_phi4_mini_server.sh`: Test script to verify the updated server
- `use_quantized_phi4_mini.py`: Script to demonstrate loading and using a quantized model
- `benchmark_phi4_mini.py`: Script to benchmark and compare different quantization methods
- `compare_memory_usage.py`: Script to compare the memory usage of different quantization methods

For more details, see the [examples README](examples/README.md).

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