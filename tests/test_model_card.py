#!/usr/bin/env python3
"""
Test script for model card generation functionality.
"""

import os
import tempfile
import unittest
from quantizer.model_card import (
    generate_model_card,
    update_model_card_with_benchmark,
    estimate_memory_usage,
    format_benchmark_results
)

class TestModelCard(unittest.TestCase):
    """Test cases for model card generation."""
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Test with known models
        self.assertAlmostEqual(estimate_memory_usage("phi-4-mini", bits=16), 8.4, delta=1.0)
        self.assertAlmostEqual(estimate_memory_usage("phi-4-mini", bits=4), 2.1, delta=0.5)
        
        # Test with model name containing parameter count
        self.assertAlmostEqual(estimate_memory_usage("llama-7b", bits=16), 14.0, delta=1.0)
        self.assertAlmostEqual(estimate_memory_usage("llama-7b", bits=4), 3.5, delta=0.5)
    
    def test_format_benchmark_results(self):
        """Test benchmark results formatting."""
        # Test with empty results
        self.assertEqual(format_benchmark_results({}), "No benchmark results available yet.")
        
        # Test with memory metrics
        results = {
            "memory_metrics": {
                "initial_memory": 0.5,
                "min_memory": 0.5,
                "max_memory": 2.0,
                "avg_memory": 1.5
            }
        }
        formatted = format_benchmark_results(results)
        self.assertIn("Memory Metrics", formatted)
        self.assertIn("Initial Memory", formatted)
        self.assertIn("0.50 GB", formatted)
        
        # Test with performance metrics
        results["performance_metrics"] = {
            "load_time": 3.5,
            "generation_tokens_per_sec": 12.5
        }
        formatted = format_benchmark_results(results)
        self.assertIn("Performance Metrics", formatted)
        self.assertIn("Load Time", formatted)
        self.assertIn("3.50 s", formatted)
        self.assertIn("12.50 tokens/s", formatted)
    
    def test_generate_model_card(self):
        """Test model card generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate a model card
            model_card_path = generate_model_card(
                model_name="test-model",
                original_model_name="original/test-model",
                method="gptq",
                bits=4,
                output_dir=temp_dir,
                group_size=128,
                desc_act=False,
                sym=True
            )
            
            # Check that the model card was created
            self.assertTrue(os.path.exists(model_card_path))
            
            # Read the model card
            with open(model_card_path, "r") as f:
                model_card = f.read()
            
            # Check that the model card contains the expected information
            self.assertIn("test-model-gptq-4bit", model_card)
            self.assertIn("original/test-model", model_card)
            self.assertIn("Group Size: 128", model_card)
            self.assertIn("Bits: 4", model_card)
            self.assertIn("Descending Activation Order: False", model_card)
            self.assertIn("Symmetric: True", model_card)
    
    def test_update_model_card_with_benchmark(self):
        """Test updating a model card with benchmark results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate a model card
            model_card_path = generate_model_card(
                model_name="test-model",
                original_model_name="original/test-model",
                method="gptq",
                bits=4,
                output_dir=temp_dir
            )
            
            # Create benchmark results
            benchmark_results = {
                "memory_metrics": {
                    "initial_memory": 0.5,
                    "min_memory": 0.5,
                    "max_memory": 2.0,
                    "avg_memory": 1.5
                },
                "performance_metrics": {
                    "load_time": 3.5,
                    "generation_tokens_per_sec": 12.5
                }
            }
            
            # Update the model card
            updated_card = update_model_card_with_benchmark(temp_dir, benchmark_results)
            
            # Check that the model card was updated
            self.assertEqual(updated_card, model_card_path)
            
            # Read the updated model card
            with open(updated_card, "r") as f:
                model_card = f.read()
            
            # Check that the model card contains the benchmark results
            self.assertIn("Memory Metrics", model_card)
            self.assertIn("Initial Memory", model_card)
            self.assertIn("Performance Metrics", model_card)
            self.assertIn("Load Time", model_card)
            self.assertIn("Generation Tokens Per Sec", model_card)

if __name__ == "__main__":
    unittest.main() 