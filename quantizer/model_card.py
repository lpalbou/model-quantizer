#!/usr/bin/env python3
"""
Module for generating model cards for quantized models.
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, Optional, Any, List, Union

# Template for the model card
MODEL_CARD_TEMPLATE = """---
language:
- en
tags:
- quantized
- {METHOD_TAG}
- {BITS}bit
license: {LICENSE}
datasets:
- {CALIBRATION_DATASET}
---

# {MODEL_NAME}-{METHOD}-{BITS}bit

This is a {BITS}-bit quantized version of [{ORIGINAL_MODEL_NAME}](https://huggingface.co/{ORIGINAL_MODEL_NAME}) using the {METHOD} quantization method.

## Model Details

- **Original Model:** [{ORIGINAL_MODEL_NAME}](https://huggingface.co/{ORIGINAL_MODEL_NAME})
- **Quantization Method:** {METHOD} ({BITS}-bit)
- **Hugging Face Transformers Compatible:** Yes
- **Quantized Date:** {DATE}
- **Quantization Parameters:**
  - Group Size: {GROUP_SIZE}
  - Bits: {BITS}
  - Descending Activation Order: {DESC_ACT}
  - Symmetric: {SYM}

## Performance Metrics

{BENCHMARK_RESULTS}

### Memory Usage
- Original Model (FP16): ~{ORIGINAL_MEMORY_USAGE} GB
- Quantized Model ({BITS}-bit): ~{QUANTIZED_MEMORY_USAGE} GB
- Memory Reduction: ~{MEMORY_REDUCTION}%

### Speed
- Load Time: {LOAD_TIME}
- Generation Speed: {GENERATION_SPEED} tokens/sec

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained("{HF_USERNAME}/{MODEL_NAME}-{METHOD}-{BITS}bit", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{HF_USERNAME}/{MODEL_NAME}-{METHOD}-{BITS}bit")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Quantization Process

This model was quantized using the [Model Quantizer](https://github.com/lpalbou/model-quantizer) tool with the following command:

```bash
model-quantizer {ORIGINAL_MODEL_NAME} --bits {BITS} --method {METHOD} --output-dir {OUTPUT_DIR} {ADDITIONAL_ARGS}
```

## License

This model is licensed under the same license as the original model: {LICENSE}.
"""

def estimate_memory_usage(model_name: str, bits: int = 16) -> float:
    """
    Estimate memory usage for a model based on its parameter count and bit width.
    
    Args:
        model_name: Name of the model
        bits: Bit width for quantization
    
    Returns:
        Estimated memory usage in GB
    """
    # This is a very rough estimation based on common model sizes
    # In a real implementation, you would want to get the actual parameter count
    model_sizes = {
        "phi-4-mini": 4.2,  # 4.2B parameters
        "llama-3-8b": 8.0,  # 8B parameters
        "llama-3-70b": 70.0,  # 70B parameters
        "gemma-2b": 2.0,  # 2B parameters
        "gemma-7b": 7.0,  # 7B parameters
        "mistral-7b": 7.0,  # 7B parameters
    }
    
    # Try to match the model name to known models
    param_count = None
    model_name_lower = model_name.lower()
    
    for key, size in model_sizes.items():
        if key in model_name_lower:
            param_count = size
            break
    
    # Default to a reasonable size if we can't determine
    if param_count is None:
        # Extract numbers from the model name
        numbers = re.findall(r'(\d+)b', model_name_lower)
        if numbers:
            param_count = float(numbers[0])
        else:
            # Default to 7B if we can't determine
            param_count = 7.0
    
    # Calculate memory usage: parameters * bytes per parameter
    bytes_per_param = bits / 8
    memory_gb = (param_count * 1e9 * bytes_per_param) / 1e9
    
    return memory_gb

def format_benchmark_results(results: Dict[str, Any]) -> str:
    """
    Format benchmark results for inclusion in the model card.
    
    Args:
        results: Dictionary containing benchmark results
    
    Returns:
        Formatted benchmark results as a string
    """
    if not results:
        return "No benchmark results available yet."
    
    formatted = "### Benchmark Results\n\n"
    
    # Add memory metrics
    if "memory_metrics" in results:
        mem = results["memory_metrics"]
        formatted += "#### Memory Metrics\n"
        formatted += "| Metric | Value |\n"
        formatted += "| ------ | ----- |\n"
        for key, value in mem.items():
            if isinstance(value, float):
                formatted += f"| {key.replace('_', ' ').title()} | {value:.2f} GB |\n"
            else:
                formatted += f"| {key.replace('_', ' ').title()} | {value} |\n"
        formatted += "\n"
    
    # Add performance metrics
    if "performance_metrics" in results:
        perf = results["performance_metrics"]
        formatted += "#### Performance Metrics\n"
        formatted += "| Metric | Value |\n"
        formatted += "| ------ | ----- |\n"
        for key, value in perf.items():
            if "time" in key and isinstance(value, float):
                formatted += f"| {key.replace('_', ' ').title()} | {value:.2f} s |\n"
            elif "tokens" in key and "per" in key and isinstance(value, float):
                formatted += f"| {key.replace('_', ' ').title()} | {value:.2f} tokens/s |\n"
            else:
                formatted += f"| {key.replace('_', ' ').title()} | {value} |\n"
        formatted += "\n"
    
    # Add quality metrics if available
    if "quality_metrics" in results:
        qual = results["quality_metrics"]
        formatted += "#### Quality Metrics\n"
        formatted += "| Metric | Original | Quantized | Difference |\n"
        formatted += "| ------ | -------- | --------- | ---------- |\n"
        for key, values in qual.items():
            if isinstance(values, dict) and "original" in values and "quantized" in values:
                diff = values["quantized"] - values["original"] if isinstance(values["quantized"], (int, float)) else "N/A"
                if isinstance(diff, (int, float)):
                    diff_str = f"{diff:+.2f}"
                else:
                    diff_str = str(diff)
                formatted += f"| {key.replace('_', ' ').title()} | {values['original']} | {values['quantized']} | {diff_str} |\n"
        formatted += "\n"
    
    # Add comparison table if available
    if "comparison" in results:
        comp = results["comparison"]
        formatted += "#### Model Comparison\n"
        formatted += "| Model | Memory | Load Time | Generation Speed | Quality |\n"
        formatted += "| ----- | ------ | --------- | ---------------- | ------- |\n"
        for model, metrics in comp.items():
            memory = f"{metrics.get('memory', 'N/A')} GB" if metrics.get('memory') else "N/A"
            load_time = f"{metrics.get('load_time', 'N/A')} s" if metrics.get('load_time') else "N/A"
            gen_speed = f"{metrics.get('generation_speed', 'N/A')} tokens/s" if metrics.get('generation_speed') else "N/A"
            quality = metrics.get('quality', 'N/A')
            formatted += f"| {model} | {memory} | {load_time} | {gen_speed} | {quality} |\n"
    
    return formatted

def generate_model_card(
    model_name: str,
    original_model_name: str,
    method: str,
    bits: int,
    output_dir: str,
    group_size: int = 128,
    desc_act: bool = False,
    sym: bool = True,
    benchmark_results: Optional[Dict[str, Any]] = None,
    hf_username: Optional[str] = None,
    calibration_dataset: str = "c4",
    license: str = "mit",
    additional_args: str = ""
) -> str:
    """
    Generate a model card for a quantized model.
    
    Args:
        model_name: Name of the quantized model
        original_model_name: Name of the original model
        method: Quantization method (gptq, bitsandbytes, awq)
        bits: Bit width for quantization
        output_dir: Directory where the model is saved
        group_size: Group size for quantization
        desc_act: Whether descending activation order was used
        sym: Whether symmetric quantization was used
        benchmark_results: Dictionary containing benchmark results
        hf_username: Hugging Face username for the model card
        calibration_dataset: Dataset used for calibration
        license: License for the model
        additional_args: Additional arguments used for quantization
    
    Returns:
        Path to the generated model card
    """
    # Format the method tag for HF
    method_tag_map = {
        "gptq": "GPTQ",
        "bitsandbytes": "BitsAndBytes",
        "bnb": "BitsAndBytes",
        "awq": "AWQ"
    }
    method_tag = method_tag_map.get(method.lower(), method.upper())
    
    # Get the current date
    date = datetime.now().strftime("%Y-%m-%d")
    
    # Estimate memory usage
    original_memory = estimate_memory_usage(original_model_name, bits=16)
    quantized_memory = estimate_memory_usage(original_model_name, bits=bits)
    memory_reduction = ((original_memory - quantized_memory) / original_memory) * 100
    
    # Format benchmark results if available
    benchmark_section = format_benchmark_results(benchmark_results) if benchmark_results else "No benchmark results available yet."
    
    # Get load time and generation speed from benchmark results if available
    load_time = "N/A"
    generation_speed = "N/A"
    if benchmark_results:
        if "performance_metrics" in benchmark_results:
            perf = benchmark_results["performance_metrics"]
            load_time = f"{perf.get('load_time', 'N/A')} s" if perf.get('load_time') else "N/A"
            generation_speed = f"{perf.get('generation_tokens_per_sec', 'N/A')} tokens/s" if perf.get('generation_tokens_per_sec') else "N/A"
    
    # Extract model name from path if needed
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    if "/" in original_model_name:
        original_model_display = original_model_name
    else:
        original_model_display = f"huggingface/{original_model_name}"
    
    # Generate the model card
    model_card = MODEL_CARD_TEMPLATE.format(
        MODEL_NAME=model_name,
        ORIGINAL_MODEL_NAME=original_model_name,
        METHOD=method,
        METHOD_TAG=method_tag,
        BITS=bits,
        DATE=date,
        GROUP_SIZE=group_size,
        DESC_ACT=desc_act,
        SYM=sym,
        BENCHMARK_RESULTS=benchmark_section,
        ORIGINAL_MEMORY_USAGE=f"{original_memory:.1f}",
        QUANTIZED_MEMORY_USAGE=f"{quantized_memory:.1f}",
        MEMORY_REDUCTION=f"{memory_reduction:.1f}",
        LOAD_TIME=load_time,
        GENERATION_SPEED=generation_speed,
        HF_USERNAME=hf_username or "YOUR_USERNAME",
        OUTPUT_DIR=output_dir,
        ADDITIONAL_ARGS=additional_args,
        LICENSE=license,
        CALIBRATION_DATASET=calibration_dataset
    )
    
    # Save the model card
    model_card_path = os.path.join(output_dir, "README.md")
    with open(model_card_path, "w") as f:
        f.write(model_card)
    
    return model_card_path

def update_model_card_with_benchmark(model_path: str, benchmark_results: Dict[str, Any]) -> str:
    """
    Update an existing model card with benchmark results.
    
    Args:
        model_path: Path to the model directory
        benchmark_results: Dictionary containing benchmark results
    
    Returns:
        Path to the updated model card
    """
    model_card_path = os.path.join(model_path, "README.md")
    
    # If the model card doesn't exist, try to create a basic one
    if not os.path.exists(model_card_path):
        # Try to extract model information from config files
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Extract model information from config
            model_name = os.path.basename(model_path)
            original_model_name = config.get("_name_or_path", "unknown")
            method = "unknown"
            bits = 0
            
            # Try to determine method and bits from model name
            if "gptq" in model_name.lower():
                method = "gptq"
            elif "bnb" in model_name.lower() or "bitsandbytes" in model_name.lower():
                method = "bitsandbytes"
            elif "awq" in model_name.lower():
                method = "awq"
            
            # Try to determine bits from model name
            bit_match = re.search(r'(\d+)bit', model_name.lower())
            if bit_match:
                bits = int(bit_match.group(1))
            else:
                # Default to 4-bit
                bits = 4
            
            # Generate a basic model card
            return generate_model_card(
                model_name=model_name,
                original_model_name=original_model_name,
                method=method,
                bits=bits,
                output_dir=model_path,
                benchmark_results=benchmark_results
            )
        else:
            # Can't determine model information, create a very basic model card
            with open(model_card_path, "w") as f:
                f.write(f"# {os.path.basename(model_path)}\n\n")
                f.write("## Performance Metrics\n\n")
                f.write(format_benchmark_results(benchmark_results))
            return model_card_path
    
    # Read the existing model card
    with open(model_card_path, "r") as f:
        model_card = f.read()
    
    # Format benchmark results
    benchmark_section = format_benchmark_results(benchmark_results)
    
    # Update the model card
    if "## Performance Metrics" in model_card:
        # Replace the existing benchmark section
        model_card = re.sub(
            r"## Performance Metrics\n\n.*?\n\n### Memory Usage",
            f"## Performance Metrics\n\n{benchmark_section}\n\n### Memory Usage",
            model_card,
            flags=re.DOTALL
        )
    else:
        # Add the benchmark section before the Usage section
        if "## Usage" in model_card:
            model_card = model_card.replace(
                "## Usage",
                f"## Performance Metrics\n\n{benchmark_section}\n\n## Usage"
            )
        else:
            # Add at the end
            model_card += f"\n\n## Performance Metrics\n\n{benchmark_section}"
    
    # Update memory usage and speed metrics if available
    if benchmark_results and "memory_metrics" in benchmark_results:
        mem = benchmark_results["memory_metrics"]
        if "initial_memory" in mem and "max_memory" in mem:
            memory_increase = mem["max_memory"] - mem["initial_memory"]
            memory_pattern = r"- Quantized Model \(.*?\): ~(.*?) GB"
            memory_match = re.search(memory_pattern, model_card)
            if memory_match:
                model_card = re.sub(
                    memory_pattern,
                    f"- Quantized Model (quantized): ~{mem['max_memory']:.1f} GB",
                    model_card
                )
    
    if benchmark_results and "performance_metrics" in benchmark_results:
        perf = benchmark_results["performance_metrics"]
        if "load_time" in perf:
            model_card = re.sub(
                r"- Load Time: .*?$",
                f"- Load Time: {perf['load_time']:.2f} s",
                model_card,
                flags=re.MULTILINE
            )
        if "generation_tokens_per_sec" in perf:
            model_card = re.sub(
                r"- Generation Speed: .*?$",
                f"- Generation Speed: {perf['generation_tokens_per_sec']:.2f} tokens/sec",
                model_card,
                flags=re.MULTILINE
            )
    
    # Write the updated model card
    with open(model_card_path, "w") as f:
        f.write(model_card)
    
    return model_card_path

def save_benchmark_results(benchmark_results: Dict[str, Any], output_path: str) -> str:
    """
    Save benchmark results to a JSON file.
    
    Args:
        benchmark_results: Dictionary containing benchmark results
        output_path: Path to save the results
    
    Returns:
        Path to the saved results
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the results
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    return output_path

def load_benchmark_results(input_path: str) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        input_path: Path to the results file
    
    Returns:
        Dictionary containing benchmark results
    """
    with open(input_path, "r") as f:
        return json.load(f) 