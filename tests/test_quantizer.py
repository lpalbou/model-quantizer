#!/usr/bin/env python3
"""
Tests for the quantizer module.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantizer import ModelQuantizer, QuantizationConfig


class TestQuantizationConfig(unittest.TestCase):
    """Tests for the QuantizationConfig class."""
    
    def test_default_config(self):
        """Test that the default configuration is valid."""
        config = QuantizationConfig()
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.method, "gptq")
        self.assertEqual(config.output_dir, "quantized-model")
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.desc_act, False)
        self.assertEqual(config.sym, True)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.use_optimum, True)
    
    def test_custom_config(self):
        """Test that a custom configuration is valid."""
        config = QuantizationConfig(
            bits=4,
            method="bitsandbytes",
            output_dir="custom-output",
            group_size=64,
            desc_act=True,
            sym=False,
            device="cpu",
            use_optimum=False
        )
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.method, "bitsandbytes")
        self.assertEqual(config.output_dir, "custom-output")
        self.assertEqual(config.group_size, 64)
        self.assertEqual(config.desc_act, True)
        self.assertEqual(config.sym, False)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.use_optimum, False)
    
    def test_invalid_bits(self):
        """Test that invalid bits raise a ValueError."""
        with self.assertRaises(ValueError):
            QuantizationConfig(bits=5)
    
    def test_invalid_method(self):
        """Test that invalid methods raise a ValueError."""
        with self.assertRaises(ValueError):
            QuantizationConfig(method="invalid")
    
    def test_to_dict(self):
        """Test that to_dict returns a valid dictionary."""
        config = QuantizationConfig()
        config_dict = config.to_dict()
        self.assertEqual(config_dict["bits"], 8)
        self.assertEqual(config_dict["method"], "gptq")
    
    def test_from_dict(self):
        """Test that from_dict creates a valid configuration."""
        config_dict = {
            "bits": 4,
            "method": "bitsandbytes",
            "output_dir": "custom-output"
        }
        config = QuantizationConfig.from_dict(config_dict)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.method, "bitsandbytes")
        self.assertEqual(config.output_dir, "custom-output")


class TestModelQuantizer(unittest.TestCase):
    """Tests for the ModelQuantizer class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test that the ModelQuantizer initializes correctly."""
        config = QuantizationConfig()
        quantizer = ModelQuantizer(config)
        self.assertEqual(quantizer.config, config)
        self.assertIsNone(quantizer.model)
        self.assertIsNone(quantizer.tokenizer)
    
    def test_get_device(self):
        """Test that _get_device returns a valid device."""
        config = QuantizationConfig(device="cpu")
        quantizer = ModelQuantizer(config)
        self.assertEqual(quantizer._get_device(), "cpu")
    
    def test_prepare_gptq_config(self):
        """Test that _prepare_gptq_config returns a valid GPTQConfig."""
        config = QuantizationConfig(bits=4, group_size=64)
        quantizer = ModelQuantizer(config)
        
        # Mock tokenizer
        class MockTokenizer:
            pass
        
        tokenizer = MockTokenizer()
        gptq_config = quantizer._prepare_gptq_config(tokenizer)
        
        self.assertEqual(gptq_config.bits, 4)
        self.assertEqual(gptq_config.group_size, 64)
    
    def test_prepare_bnb_config(self):
        """Test that _prepare_bnb_config returns a valid BitsAndBytesConfig."""
        config = QuantizationConfig(bits=8, method="bitsandbytes")
        quantizer = ModelQuantizer(config)
        bnb_config = quantizer._prepare_bnb_config()
        
        self.assertTrue(bnb_config.load_in_8bit)
        
        config = QuantizationConfig(bits=4, method="bitsandbytes")
        quantizer = ModelQuantizer(config)
        bnb_config = quantizer._prepare_bnb_config()
        
        self.assertTrue(bnb_config.load_in_4bit)
    
    def test_invalid_bnb_bits(self):
        """Test that invalid bits for BitsAndBytes raise a ValueError."""
        config = QuantizationConfig(bits=2, method="bitsandbytes")
        quantizer = ModelQuantizer(config)
        
        with self.assertRaises(ValueError):
            quantizer._prepare_bnb_config()


if __name__ == "__main__":
    unittest.main() 