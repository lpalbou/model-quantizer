#!/usr/bin/env python3
"""
Benchmark script to compare original and quantized models.
Measures and reports:
- Memory usage (min, avg, max)
- Loading time
- Inference speed for different prompt lengths
- Quality metrics (optional)
"""

import os
import time
import json
import argparse
import psutil
import torch
import numpy as np
from datetime import datetime
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Import model card utilities
try:
    from quantizer.model_card import update_model_card_with_benchmark
    MODEL_CARD_AVAILABLE = True
except ImportError:
    MODEL_CARD_AVAILABLE = False

class MemoryTracker:
    """Track memory usage during model interaction."""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.min_memory = float('inf')
        self.max_memory = 0
        self.memory_samples = []
        self.initial_memory = self.get_current_memory()
        self.memory_samples.append(self.initial_memory)
    
    def get_current_memory(self):
        """Get current memory usage in GB."""
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024 ** 3)  # Convert to GB
    
    def update(self):
        """Update memory statistics."""
        current = self.get_current_memory()
        self.min_memory = min(self.min_memory, current)
        self.max_memory = max(self.max_memory, current)
        self.memory_samples.append(current)
        return current
    
    def get_stats(self):
        """Get memory statistics."""
        return {
            "min_memory": self.min_memory,
            "max_memory": self.max_memory,
            "avg_memory": np.mean(self.memory_samples),
            "std_memory": np.std(self.memory_samples),
            "current_memory": self.get_current_memory(),
            "initial_memory": self.initial_memory,
            "memory_increase": self.get_current_memory() - self.initial_memory
        }

class PerformanceTracker:
    """Track performance metrics during model interaction."""
    def __init__(self):
        self.start_time = time.time()
        self.load_duration = 0
        self.prompt_eval_count = 0
        self.prompt_eval_duration = 0
        self.eval_count = 0
        self.eval_duration = 0
        self.generation_samples = []
    
    def set_load_duration(self, duration):
        """Set the model loading duration."""
        self.load_duration = duration
    
    def add_prompt_eval(self, token_count, duration):
        """Add prompt evaluation metrics."""
        self.prompt_eval_count += token_count
        self.prompt_eval_duration += duration
    
    def add_generation(self, token_count, duration):
        """Add generation metrics."""
        self.eval_count += token_count
        self.eval_duration += duration
        if token_count > 0 and duration > 0:
            self.generation_samples.append(token_count / duration)
    
    def get_stats(self):
        """Get performance statistics."""
        total_duration = time.time() - self.start_time
        prompt_eval_rate = self.prompt_eval_count / self.prompt_eval_duration if self.prompt_eval_duration > 0 else 0
        eval_rate = self.eval_count / self.eval_duration if self.eval_duration > 0 else 0
        
        stats = {
            "total_duration": total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": self.prompt_eval_duration,
            "prompt_eval_rate": prompt_eval_rate,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
            "eval_rate": eval_rate,
        }
        
        if self.generation_samples:
            stats.update({
                "min_generation_rate": min(self.generation_samples),
                "max_generation_rate": max(self.generation_samples),
                "avg_generation_rate": np.mean(self.generation_samples),
                "std_generation_rate": np.std(self.generation_samples)
            })
        
        return stats

class ModelBenchmark:
    """Benchmark a model's performance."""
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.memory_tracker = MemoryTracker()
        self.perf_tracker = PerformanceTracker()
        self.model = None
        self.tokenizer = None
        self.model_name = os.path.basename(model_path)
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with performance tracking."""
        print(f"\n{'=' * 50}")
        print(f"Loading model: {self.model_name}")
        print(f"{'=' * 50}")
        print(f"Initial memory usage: {self.memory_tracker.get_current_memory():.2f} GB")
        
        # Set environment variables for memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Measure loading time
        start_time = time.time()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        
        load_time = time.time() - start_time
        self.perf_tracker.set_load_duration(load_time)
        
        current_memory = self.memory_tracker.update()
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Current memory usage: {current_memory:.2f} GB")
        print(f"Memory increase: {current_memory - self.memory_tracker.initial_memory:.2f} GB")
    
    def generate_text(self, prompt, max_new_tokens=100, verbose=True):
        """Generate text from a prompt, measuring time and performance."""
        if verbose:
            print(f"\nGenerating text for prompt: '{prompt[:50]}...' ({len(prompt)} chars)")
        
        # Update memory stats
        self.memory_tracker.update()
        
        # Tokenize prompt
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        prompt_tokens = len(input_ids[0])
        tokenize_time = time.time() - start_time
        self.perf_tracker.add_prompt_eval(prompt_tokens, tokenize_time)
        
        if verbose:
            print(f"Tokenized {prompt_tokens} tokens in {tokenize_time:.4f} seconds")
        
        # Set up streamer for token-by-token generation
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        # Start generation
        start_time = time.time()
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Collect generated text
        generated_text = ""
        for text in streamer:
            generated_text += text
            if verbose:
                print(text, end="", flush=True)
        
        # Update performance stats
        generation_time = time.time() - start_time
        generated_tokens = len(self.tokenizer.encode(generated_text)) - prompt_tokens
        self.perf_tracker.add_generation(generated_tokens, generation_time)
        
        # Update memory stats
        self.memory_tracker.update()
        
        if verbose:
            print(f"\n\nGeneration completed in {generation_time:.2f} seconds")
            print(f"Generated {generated_tokens} tokens at {generated_tokens/generation_time:.2f} tokens/second")
        
        return {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "tokenize_time": tokenize_time,
            "generation_time": generation_time,
            "tokens_per_second": generated_tokens / generation_time if generation_time > 0 else 0
        }
    
    def run_benchmark(self, prompts, max_new_tokens=100, verbose=True):
        """Run benchmark on a set of prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"\nPrompt {i+1}/{len(prompts)}")
            
            result = self.generate_text(prompt["text"], max_new_tokens, verbose)
            result["prompt_name"] = prompt["name"]
            result["prompt_category"] = prompt["category"]
            results.append(result)
        
        return results
    
    def get_stats(self):
        """Get combined statistics."""
        return {
            "memory": self.memory_tracker.get_stats(),
            "performance": self.perf_tracker.get_stats()
        }
    
    def cleanup(self):
        """Clean up resources."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def create_test_prompts():
    """Create a set of test prompts with different lengths and categories."""
    prompts = []
    
    # Short prompts (< 100 tokens)
    prompts.extend([
        {
            "name": "short_factual",
            "category": "short",
            "text": "What is the capital of France?"
        },
        {
            "name": "short_creative",
            "category": "short",
            "text": "Write a haiku about artificial intelligence."
        },
        {
            "name": "short_reasoning",
            "category": "short",
            "text": "If a train travels at 60 mph, how far will it go in 2.5 hours?"
        }
    ])
    
    # Medium prompts (100-500 tokens)
    prompts.extend([
        {
            "name": "medium_factual",
            "category": "medium",
            "text": "Explain quantum computing in simple terms. Include the concepts of qubits, superposition, and entanglement. Also briefly describe potential applications of quantum computing."
        },
        {
            "name": "medium_creative",
            "category": "medium",
            "text": "Write a short story about a robot that develops consciousness. The story should have a beginning, middle, and end, with at least one character besides the robot. Include dialogue and a surprising twist at the end."
        },
        {
            "name": "medium_reasoning",
            "category": "medium",
            "text": "A company produces widgets at a cost of $5 per unit. Fixed costs are $1000 per month. Each widget sells for $8. What is the break-even point in units per month? If the company wants to make a profit of $2000 per month, how many widgets must they sell? Show your work and explain your reasoning."
        }
    ])
    
    # Long prompts (500-1000 tokens)
    long_text = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

    The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

    Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when obtaining funding for AI projects was difficult.

    In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began.

    Interest in neural networks and "connectionism" was revived by David Rumelhart and others in the middle of the 1980s. Artificial neural networks are a type of machine learning which had been studied since the 1960s, but the field went through a renaissance in the 1980s after the development of the backpropagation algorithm. Neural networks became commercially viable in the 1990s as a result of this renewed interest.

    The development of metal–oxide–semiconductor (MOS) very-large-scale integration (VLSI), in the form of complementary MOS (CMOS) technology, enabled the development of practical artificial neural networks in the 1980s. Deep learning began to show significant results in the 2000s, and in the early 2010s, AI once again began to flourish. This was helped by three key factors: the vast amounts of data that was suddenly available, faster computers, and better algorithms.

    In the 21st century, AI techniques have experienced a resurgence following concurrent advances in computer power, large amounts of data, and theoretical understanding. AI techniques have become an essential part of the technology industry, helping to solve many challenging problems in computer science, software engineering and operations research.
    """
    
    prompts.extend([
        {
            "name": "long_factual",
            "category": "long",
            "text": f"Summarize the following text about the history of AI: {long_text}"
        },
        {
            "name": "long_creative",
            "category": "long",
            "text": f"Rewrite the following text as a dialogue between a professor and a student: {long_text}"
        },
        {
            "name": "long_reasoning",
            "category": "long",
            "text": f"Analyze the following text about AI history. Identify key turning points, technological breakthroughs, and funding patterns. What lessons can we learn from this history that might apply to current AI development? {long_text}"
        }
    ])
    
    # Very long prompts (1000+ tokens) - for testing context window handling
    very_long_text = long_text * 3  # Repeat the long text to make it even longer
    
    prompts.extend([
        {
            "name": "very_long_factual",
            "category": "very_long",
            "text": f"Extract the 5 most important facts from this text: {very_long_text}"
        },
        {
            "name": "very_long_creative",
            "category": "very_long",
            "text": f"Write a poem inspired by the themes in this text: {very_long_text}"
        }
    ])
    
    return prompts

def compare_models(original_model_path, quantized_model_path, device="cpu", max_new_tokens=100, verbose=True, output_file=None, update_model_card=False):
    """Compare original and quantized models."""
    prompts = create_test_prompts()
    
    # Benchmark original model
    print("\n" + "=" * 80)
    print(f"BENCHMARKING ORIGINAL MODEL: {original_model_path}")
    print("=" * 80)
    original_benchmark = ModelBenchmark(original_model_path, device)
    original_results = original_benchmark.run_benchmark(prompts, max_new_tokens, verbose)
    original_stats = original_benchmark.get_stats()
    original_benchmark.cleanup()
    
    # Benchmark quantized model
    print("\n" + "=" * 80)
    print(f"BENCHMARKING QUANTIZED MODEL: {quantized_model_path}")
    print("=" * 80)
    quantized_benchmark = ModelBenchmark(quantized_model_path, device)
    quantized_results = quantized_benchmark.run_benchmark(prompts, max_new_tokens, verbose)
    quantized_stats = quantized_benchmark.get_stats()
    quantized_benchmark.cleanup()
    
    # Combine results
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "original_model": original_model_path,
            "quantized_model": quantized_model_path,
            "device": device,
            "max_new_tokens": max_new_tokens
        },
        "original": {
            "results": original_results,
            "stats": original_stats
        },
        "quantized": {
            "results": quantized_results,
            "stats": quantized_stats
        }
    }
    
    # Save results if output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\nBenchmark results saved to {output_file}")
    
    # Update model card if requested
    if update_model_card and MODEL_CARD_AVAILABLE:
        try:
            # Format benchmark results for model card
            formatted_results = {
                "memory_metrics": {
                    "initial_memory": quantized_stats["memory"]["initial_memory"],
                    "min_memory": quantized_stats["memory"]["min_memory"],
                    "max_memory": quantized_stats["memory"]["max_memory"],
                    "avg_memory": quantized_stats["memory"]["avg_memory"]
                },
                "performance_metrics": {
                    "load_time": quantized_stats["performance"]["load_time"],
                    "prompt_tokens_per_sec": quantized_stats["performance"]["prompt_tokens_per_sec"],
                    "generation_tokens_per_sec": quantized_stats["performance"]["generation_tokens_per_sec"]
                },
                "quality_metrics": {},
                "comparison": {
                    "Original": {
                        "memory": original_stats["memory"]["max_memory"],
                        "load_time": original_stats["performance"]["load_time"],
                        "generation_speed": original_stats["performance"]["generation_tokens_per_sec"],
                        "quality": "Baseline"
                    },
                    "Quantized": {
                        "memory": quantized_stats["memory"]["max_memory"],
                        "load_time": quantized_stats["performance"]["load_time"],
                        "generation_speed": quantized_stats["performance"]["generation_tokens_per_sec"],
                        "quality": "See metrics"
                    }
                }
            }
            
            # Add quality metrics if available
            if "quality" in quantized_stats and "quality" in original_stats:
                for metric, value in quantized_stats["quality"].items():
                    formatted_results["quality_metrics"][metric] = {
                        "original": original_stats["quality"].get(metric, 0),
                        "quantized": value
                    }
            
            # Update the model card
            updated_card = update_model_card_with_benchmark(quantized_model_path, formatted_results)
            print(f"\nModel card updated with benchmark results: {updated_card}")
        except Exception as e:
            print(f"\nFailed to update model card: {e}")
    
    # Print summary
    print_benchmark_summary(benchmark_results)
    
    return benchmark_results

def print_benchmark_summary(results):
    """Print a summary of benchmark results."""
    original_model = os.path.basename(results["config"]["original_model"])
    quantized_model = os.path.basename(results["config"]["quantized_model"])
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Memory usage
    print("\nMEMORY USAGE (GB):")
    print(f"{'Metric':<15} {'Original':<15} {'Quantized':<15} {'Difference':<15} {'Ratio':<10}")
    print("-" * 70)
    
    orig_mem = results["original"]["stats"]["memory"]
    quant_mem = results["quantized"]["stats"]["memory"]
    
    metrics = ["initial_memory", "min_memory", "max_memory", "avg_memory", "memory_increase"]
    for metric in metrics:
        orig_val = orig_mem[metric]
        quant_val = quant_mem[metric]
        diff = quant_val - orig_val
        ratio = quant_val / orig_val if orig_val != 0 else float('inf')
        
        print(f"{metric:<15} {orig_val:<15.2f} {quant_val:<15.2f} {diff:<15.2f} {ratio:<10.2f}")
    
    # Loading performance
    print("\nLOADING PERFORMANCE:")
    orig_perf = results["original"]["stats"]["performance"]
    quant_perf = results["quantized"]["stats"]["performance"]
    
    orig_load = orig_perf["load_duration"]
    quant_load = quant_perf["load_duration"]
    load_diff = quant_load - orig_load
    load_ratio = quant_load / orig_load if orig_load != 0 else float('inf')
    
    print(f"{'Metric':<15} {'Original':<15} {'Quantized':<15} {'Difference':<15} {'Ratio':<10}")
    print("-" * 70)
    print(f"{'Load time (s)':<15} {orig_load:<15.2f} {quant_load:<15.2f} {load_diff:<15.2f} {load_ratio:<10.2f}")
    
    # Generation performance by category
    print("\nGENERATION PERFORMANCE BY PROMPT CATEGORY:")
    
    # Group results by category
    categories = {}
    for result in results["original"]["results"]:
        category = result["prompt_category"]
        if category not in categories:
            categories[category] = {"original": [], "quantized": []}
        categories[category]["original"].append(result)
    
    for result in results["quantized"]["results"]:
        category = result["prompt_category"]
        categories[category]["quantized"].append(result)
    
    # Print performance by category
    for category, data in categories.items():
        print(f"\n{category.upper()} PROMPTS:")
        print(f"{'Metric':<20} {'Original':<15} {'Quantized':<15} {'Difference':<15} {'Ratio':<10}")
        print("-" * 75)
        
        # Calculate averages
        orig_gen_time = np.mean([r["generation_time"] for r in data["original"]])
        quant_gen_time = np.mean([r["generation_time"] for r in data["quantized"]])
        gen_time_diff = quant_gen_time - orig_gen_time
        gen_time_ratio = quant_gen_time / orig_gen_time if orig_gen_time != 0 else float('inf')
        
        orig_tokens_per_sec = np.mean([r["tokens_per_second"] for r in data["original"]])
        quant_tokens_per_sec = np.mean([r["tokens_per_second"] for r in data["quantized"]])
        tokens_per_sec_diff = quant_tokens_per_sec - orig_tokens_per_sec
        tokens_per_sec_ratio = quant_tokens_per_sec / orig_tokens_per_sec if orig_tokens_per_sec != 0 else float('inf')
        
        print(f"{'Generation time (s)':<20} {orig_gen_time:<15.2f} {quant_gen_time:<15.2f} {gen_time_diff:<15.2f} {gen_time_ratio:<10.2f}")
        print(f"{'Tokens per second':<20} {orig_tokens_per_sec:<15.2f} {quant_tokens_per_sec:<15.2f} {tokens_per_sec_diff:<15.2f} {tokens_per_sec_ratio:<10.2f}")
    
    # Overall generation performance
    print("\nOVERALL GENERATION PERFORMANCE:")
    print(f"{'Metric':<20} {'Original':<15} {'Quantized':<15} {'Difference':<15} {'Ratio':<10}")
    print("-" * 75)
    
    metrics = [
        ("prompt_eval_count", "Prompt tokens"),
        ("prompt_eval_duration", "Prompt eval time (s)"),
        ("prompt_eval_rate", "Prompt tokens/sec"),
        ("eval_count", "Generated tokens"),
        ("eval_duration", "Generation time (s)"),
        ("eval_rate", "Generation tokens/sec")
    ]
    
    for key, label in metrics:
        orig_val = orig_perf[key]
        quant_val = quant_perf[key]
        diff = quant_val - orig_val
        ratio = quant_val / orig_val if orig_val != 0 else float('inf')
        
        print(f"{label:<20} {orig_val:<15.2f} {quant_val:<15.2f} {diff:<15.2f} {ratio:<10.2f}")
    
    # Summary
    print("\nSUMMARY:")
    memory_reduction = (1 - (quant_mem["max_memory"] / orig_mem["max_memory"])) * 100
    speed_change = ((quant_perf["eval_rate"] / orig_perf["eval_rate"]) - 1) * 100
    
    print(f"Memory reduction: {memory_reduction:.2f}%")
    print(f"Generation speed change: {speed_change:.2f}%")
    
    if speed_change > 0:
        print(f"The quantized model is {abs(speed_change):.2f}% faster than the original model")
    else:
        print(f"The quantized model is {abs(speed_change):.2f}% slower than the original model")

def main():
    parser = argparse.ArgumentParser(description="Benchmark original and quantized models")
    parser.add_argument("--original", type=str, required=True, help="Path to the original model")
    parser.add_argument("--quantized", type=str, required=True, help="Path to the quantized model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on (cpu, cuda, mps)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--output", type=str, help="Output file to save benchmark results (JSON)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--update-model-card", action="store_true", help="Update the model card with benchmark results")
    args = parser.parse_args()
    
    compare_models(
        args.original,
        args.quantized,
        args.device,
        args.max_new_tokens,
        not args.quiet,
        args.output,
        args.update_model_card
    )

if __name__ == "__main__":
    main() 