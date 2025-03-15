#!/usr/bin/env python3
"""
Tests for the QuantizationConfig class.
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
from unittest.mock import patch, mock_open

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantizer.quantization_config import QuantizationConfig


class TestQuantizationConfig(unittest.TestCase):
    """Tests for the QuantizationConfig class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_default(self):
        """Test that QuantizationConfig initializes with default values."""
        config = QuantizationConfig()
        
        # Check default values
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.method, "gptq")
        self.assertEqual(config.output_dir, "quantized_model")
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.desc_act, False)
        self.assertEqual(config.sym, False)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.use_optimum, False)
    
    def test_init_custom(self):
        """Test that QuantizationConfig initializes with custom values."""
        config = QuantizationConfig(
            bits=4,
            method="bitsandbytes",
            output_dir=self.temp_dir,
            group_size=64,
            desc_act=True,
            sym=True,
            device="cuda",
            use_optimum=True
        )
        
        # Check custom values
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.method, "bitsandbytes")
        self.assertEqual(config.output_dir, self.temp_dir)
        self.assertEqual(config.group_size, 64)
        self.assertEqual(config.desc_act, True)
        self.assertEqual(config.sym, True)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.use_optimum, True)
    
    def test_post_init_valid_method(self):
        """Test that __post_init__ validates the method."""
        # Valid methods should not raise an error
        QuantizationConfig(method="gptq")
        QuantizationConfig(method="bitsandbytes")
        QuantizationConfig(method="awq")
        
        # Invalid method should raise a ValueError
        with self.assertRaises(ValueError):
            QuantizationConfig(method="invalid_method")
    
    def test_post_init_valid_bits(self):
        """Test that __post_init__ validates the bits."""
        # Valid bit widths should not raise an error
        QuantizationConfig(bits=2)
        QuantizationConfig(bits=3)
        QuantizationConfig(bits=4)
        QuantizationConfig(bits=8)
        
        # Invalid bit width should raise a ValueError
        with self.assertRaises(ValueError):
            QuantizationConfig(bits=5)
    
    @patch("logging.warning")
    def test_post_init_warning_bits(self, mock_warning):
        """Test that __post_init__ warns about non-recommended bit widths."""
        # 2-bit and 3-bit should trigger a warning for GPTQ
        QuantizationConfig(bits=2, method="gptq")
        mock_warning.assert_called_with(
            "2-bit quantization is not recommended for GPTQ. "
            "Recommended bit widths are 4 and 8."
        )
        
        mock_warning.reset_mock()
        QuantizationConfig(bits=3, method="gptq")
        mock_warning.assert_called_with(
            "3-bit quantization is not recommended for GPTQ. "
            "Recommended bit widths are 4 and 8."
        )
        
        # 4-bit and 8-bit should not trigger a warning for GPTQ
        mock_warning.reset_mock()
        QuantizationConfig(bits=4, method="gptq")
        mock_warning.assert_not_called()
        
        mock_warning.reset_mock()
        QuantizationConfig(bits=8, method="gptq")
        mock_warning.assert_not_called()
    
    def test_to_dict(self):
        """Test that to_dict returns a dictionary with the correct values."""
        config = QuantizationConfig(
            bits=4,
            method="gptq",
            output_dir=self.temp_dir,
            group_size=64,
            desc_act=True,
            sym=True,
            device="cuda",
            use_optimum=True
        )
        
        config_dict = config.to_dict()
        
        # Check that the dictionary has the correct values
        self.assertEqual(config_dict["bits"], 4)
        self.assertEqual(config_dict["method"], "gptq")
        self.assertEqual(config_dict["output_dir"], self.temp_dir)
        self.assertEqual(config_dict["group_size"], 64)
        self.assertEqual(config_dict["desc_act"], True)
        self.assertEqual(config_dict["sym"], True)
        self.assertEqual(config_dict["device"], "cuda")
        self.assertEqual(config_dict["use_optimum"], True)
    
    def test_from_dict(self):
        """Test that from_dict creates a QuantizationConfig with the correct values."""
        config_dict = {
            "bits": 4,
            "method": "gptq",
            "output_dir": self.temp_dir,
            "group_size": 64,
            "desc_act": True,
            "sym": True,
            "device": "cuda",
            "use_optimum": True
        }
        
        config = QuantizationConfig.from_dict(config_dict)
        
        # Check that the config has the correct values
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.method, "gptq")
        self.assertEqual(config.output_dir, self.temp_dir)
        self.assertEqual(config.group_size, 64)
        self.assertEqual(config.desc_act, True)
        self.assertEqual(config.sym, True)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.use_optimum, True)
    
    def test_save(self):
        """Test that save writes the config to a file."""
        config = QuantizationConfig(
            bits=4,
            method="gptq",
            output_dir=self.temp_dir,
            group_size=64,
            desc_act=True,
            sym=True,
            device="cuda",
            use_optimum=True
        )
        
        # Mock open to capture the written data
        with patch("builtins.open", mock_open()) as mock_file:
            config.save(self.config_path)
            
            # Check that the file was opened for writing
            mock_file.assert_called_once_with(self.config_path, "w")
            
            # Check that the correct data was written
            written_data = mock_file().write.call_args[0][0]
            config_dict = json.loads(written_data)
            self.assertEqual(config_dict["bits"], 4)
            self.assertEqual(config_dict["method"], "gptq")
            self.assertEqual(config_dict["output_dir"], self.temp_dir)
            self.assertEqual(config_dict["group_size"], 64)
            self.assertEqual(config_dict["desc_act"], True)
            self.assertEqual(config_dict["sym"], True)
            self.assertEqual(config_dict["device"], "cuda")
            self.assertEqual(config_dict["use_optimum"], True)
    
    def test_load(self):
        """Test that load reads the config from a file."""
        config_dict = {
            "bits": 4,
            "method": "gptq",
            "output_dir": self.temp_dir,
            "group_size": 64,
            "desc_act": True,
            "sym": True,
            "device": "cuda",
            "use_optimum": True
        }
        
        # Mock open to return the config data
        with patch("builtins.open", mock_open(read_data=json.dumps(config_dict))):
            config = QuantizationConfig.load(self.config_path)
            
            # Check that the config has the correct values
            self.assertEqual(config.bits, 4)
            self.assertEqual(config.method, "gptq")
            self.assertEqual(config.output_dir, self.temp_dir)
            self.assertEqual(config.group_size, 64)
            self.assertEqual(config.desc_act, True)
            self.assertEqual(config.sym, True)
            self.assertEqual(config.device, "cuda")
            self.assertEqual(config.use_optimum, True)
    
    def test_str(self):
        """Test that __str__ returns a string representation of the config."""
        config = QuantizationConfig(
            bits=4,
            method="gptq",
            output_dir=self.temp_dir,
            group_size=64,
            desc_act=True,
            sym=True,
            device="cuda",
            use_optimum=True
        )
        
        config_str = str(config)
        
        # Check that the string contains all the config values
        self.assertIn("bits=4", config_str)
        self.assertIn(f"method='gptq'", config_str)
        self.assertIn(f"output_dir='{self.temp_dir}'", config_str)
        self.assertIn("group_size=64", config_str)
        self.assertIn("desc_act=True", config_str)
        self.assertIn("sym=True", config_str)
        self.assertIn("device='cuda'", config_str)
        self.assertIn("use_optimum=True", config_str)


if __name__ == "__main__":
    unittest.main() 