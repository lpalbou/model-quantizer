#!/usr/bin/env python3
"""
Tests for the ModelQuantizer class.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
import json
import torch

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantizer.model_quantizer import ModelQuantizer
from quantizer.quantization_config import QuantizationConfig


class TestModelQuantizer(unittest.TestCase):
    """Tests for the ModelQuantizer class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = QuantizationConfig(
            bits=4,
            method="gptq",
            output_dir=self.temp_dir,
            group_size=64,
            desc_act=True,
            sym=False,
            device="cpu",
            use_optimum=False
        )
        self.model_name = "test/model"
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test that the ModelQuantizer initializes correctly."""
        quantizer = ModelQuantizer(self.config)
        self.assertEqual(quantizer.config, self.config)
        self.assertIsNone(quantizer.model)
        self.assertIsNone(quantizer.tokenizer)
    
    def test_init_default_config(self):
        """Test that the ModelQuantizer initializes with default config."""
        quantizer = ModelQuantizer()
        self.assertIsInstance(quantizer.config, QuantizationConfig)
        self.assertEqual(quantizer.config.bits, 8)
        self.assertEqual(quantizer.config.method, "gptq")
    
    @patch("quantizer.model_quantizer.platform")
    def test_check_platform_compatibility_macos_bitsandbytes(self, mock_platform):
        """Test platform compatibility check for macOS with BitsAndBytes."""
        # Mock platform.system to return "Darwin" (macOS)
        mock_platform.system.return_value = "Darwin"
        
        # Create config with BitsAndBytes
        config = QuantizationConfig(method="bitsandbytes")
        
        # Mock stdin/stdout to simulate non-interactive mode
        with patch("sys.stdin.isatty", return_value=False), \
             patch("sys.stdout.isatty", return_value=False):
            quantizer = ModelQuantizer(config)
            # Method should still be bitsandbytes since we're in non-interactive mode
            self.assertEqual(quantizer.config.method, "bitsandbytes")
    
    @patch("quantizer.model_quantizer.platform")
    @patch("builtins.input", return_value="n")
    def test_check_platform_compatibility_macos_bitsandbytes_interactive(self, mock_input, mock_platform):
        """Test platform compatibility check for macOS with BitsAndBytes in interactive mode."""
        # Mock platform.system to return "Darwin" (macOS)
        mock_platform.system.return_value = "Darwin"
        
        # Create config with BitsAndBytes
        config = QuantizationConfig(method="bitsandbytes")
        
        # Mock stdin/stdout to simulate interactive mode
        with patch("sys.stdin.isatty", return_value=True), \
             patch("sys.stdout.isatty", return_value=True):
            quantizer = ModelQuantizer(config)
            # Method should be switched to gptq since we answered "n" to the prompt
            self.assertEqual(quantizer.config.method, "gptq")
    
    @patch("quantizer.model_quantizer.platform")
    def test_check_platform_compatibility_macos_mps(self, mock_platform):
        """Test platform compatibility check for macOS with MPS."""
        # Mock platform.system to return "Darwin" (macOS)
        mock_platform.system.return_value = "Darwin"
        
        # Create config with MPS device
        config = QuantizationConfig(device="mps")
        
        # Mock environment variables
        with patch.dict(os.environ, {}, clear=True):
            quantizer = ModelQuantizer(config)
            # PYTORCH_ENABLE_MPS_FALLBACK should be set
            self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")
    
    @patch("torch.cuda.is_available", return_value=True)
    def test_get_device_cuda(self, mock_cuda_available):
        """Test that _get_device returns 'cuda' when available."""
        config = QuantizationConfig(device="auto")
        quantizer = ModelQuantizer(config)
        self.assertEqual(quantizer._get_device(), "cuda")
    
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_get_device_mps(self, mock_mps_available, mock_cuda_available):
        """Test that _get_device returns 'mps' when available."""
        config = QuantizationConfig(device="auto")
        quantizer = ModelQuantizer(config)
        self.assertEqual(quantizer._get_device(), "mps")
    
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_get_device_cpu(self, mock_mps_available, mock_cuda_available):
        """Test that _get_device returns 'cpu' when no accelerator is available."""
        config = QuantizationConfig(device="auto")
        quantizer = ModelQuantizer(config)
        self.assertEqual(quantizer._get_device(), "cpu")
    
    def test_prepare_gptq_config(self):
        """Test that _prepare_gptq_config returns a valid GPTQConfig."""
        quantizer = ModelQuantizer(self.config)
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        
        # Call _prepare_gptq_config
        gptq_config = quantizer._prepare_gptq_config(mock_tokenizer)
        
        # Check that the config has the correct values
        self.assertEqual(gptq_config.bits, 4)
        self.assertEqual(gptq_config.group_size, 64)
        self.assertEqual(gptq_config.desc_act, True)
        self.assertEqual(gptq_config.sym, False)
    
    def test_prepare_bnb_config_8bit(self):
        """Test that _prepare_bnb_config returns a valid 8-bit BitsAndBytesConfig."""
        config = QuantizationConfig(bits=8, method="bitsandbytes")
        quantizer = ModelQuantizer(config)
        
        # Call _prepare_bnb_config
        bnb_config = quantizer._prepare_bnb_config()
        
        # Check that the config has the correct values
        self.assertTrue(bnb_config.load_in_8bit)
        self.assertFalse(hasattr(bnb_config, "load_in_4bit"))
    
    def test_prepare_bnb_config_4bit(self):
        """Test that _prepare_bnb_config returns a valid 4-bit BitsAndBytesConfig."""
        config = QuantizationConfig(bits=4, method="bitsandbytes")
        quantizer = ModelQuantizer(config)
        
        # Call _prepare_bnb_config
        bnb_config = quantizer._prepare_bnb_config()
        
        # Check that the config has the correct values
        self.assertTrue(bnb_config.load_in_4bit)
        self.assertFalse(hasattr(bnb_config, "load_in_8bit"))
    
    def test_prepare_bnb_config_invalid_bits(self):
        """Test that _prepare_bnb_config raises an error for invalid bit widths."""
        config = QuantizationConfig(bits=2, method="bitsandbytes")
        quantizer = ModelQuantizer(config)
        
        # Check that calling _prepare_bnb_config raises a ValueError
        with self.assertRaises(ValueError):
            quantizer._prepare_bnb_config()
    
    @patch("quantizer.model_quantizer.AutoModelForCausalLM")
    @patch("quantizer.model_quantizer.AutoTokenizer")
    def test_quantize_gptq(self, mock_tokenizer, mock_model):
        """Test quantizing a model with GPTQ."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Create quantizer
        quantizer = ModelQuantizer(self.config)
        
        # Call quantize
        model, tokenizer = quantizer.quantize(self.model_name)
        
        # Check that the tokenizer was loaded
        mock_tokenizer.from_pretrained.assert_called_once_with(self.model_name)
        
        # Check that the model was loaded with the correct parameters
        mock_model.from_pretrained.assert_called_once()
        call_args = mock_model.from_pretrained.call_args[0]
        call_kwargs = mock_model.from_pretrained.call_args[1]
        self.assertEqual(call_args[0], self.model_name)
        self.assertEqual(call_kwargs["device_map"], "cpu")
        self.assertIsNotNone(call_kwargs["quantization_config"])
        self.assertEqual(call_kwargs["quantization_config"].bits, 4)
        
        # Check that the model and tokenizer were set
        self.assertEqual(quantizer.model, mock_model_instance)
        self.assertEqual(quantizer.tokenizer, mock_tokenizer_instance)
        
        # Check that the returned model and tokenizer are correct
        self.assertEqual(model, mock_model_instance)
        self.assertEqual(tokenizer, mock_tokenizer_instance)
    
    @patch("quantizer.model_quantizer.AutoModelForCausalLM")
    @patch("quantizer.model_quantizer.AutoTokenizer")
    def test_save(self, mock_tokenizer, mock_model):
        """Test saving a quantized model."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        
        # Create quantizer
        quantizer = ModelQuantizer(self.config)
        quantizer.model = mock_model_instance
        quantizer.tokenizer = mock_tokenizer_instance
        
        # Call save
        quantizer.save()
        
        # Check that the model and tokenizer were saved
        mock_model_instance.save_pretrained.assert_called_once_with(self.temp_dir)
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(self.temp_dir)
    
    def test_save_no_model(self):
        """Test that save raises an error if no model is loaded."""
        quantizer = ModelQuantizer(self.config)
        with self.assertRaises(ValueError):
            quantizer.save()
    
    @patch("quantizer.model_quantizer.create_repo")
    @patch("builtins.open", new_callable=mock_open)
    def test_publish_to_hub(self, mock_file, mock_create_repo):
        """Test publishing a model to the Hugging Face Hub."""
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Create quantizer
        quantizer = ModelQuantizer(self.config)
        quantizer.model = mock_model
        quantizer.tokenizer = mock_tokenizer
        
        # Add model_name to config for model card
        quantizer.config.model_name = self.model_name
        
        # Call publish_to_hub
        repo_id = "username/model"
        url = quantizer.publish_to_hub(repo_id)
        
        # Check that the repository was created
        mock_create_repo.assert_called_once_with(repo_id, private=False, exist_ok=True)
        
        # Check that the model and tokenizer were pushed
        mock_model.push_to_hub.assert_called_once_with(repo_id, commit_message="Upload quantized model")
        mock_tokenizer.push_to_hub.assert_called_once_with(repo_id, commit_message="Upload quantized model")
        
        # Check that the returned URL is correct
        self.assertEqual(url, f"https://huggingface.co/{repo_id}")
    
    def test_publish_to_hub_no_model(self):
        """Test that publish_to_hub raises an error if no model is loaded."""
        quantizer = ModelQuantizer(self.config)
        with self.assertRaises(ValueError):
            quantizer.publish_to_hub("username/model")
    
    @patch("quantizer.model_quantizer.AutoModelForCausalLM")
    @patch("quantizer.model_quantizer.AutoTokenizer")
    def test_load(self, mock_tokenizer, mock_model):
        """Test loading a quantized model."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock the config file
        config_dict = self.config.to_dict()
        with patch("builtins.open", mock_open(read_data=json.dumps(config_dict))), \
             patch("os.path.exists", return_value=True):
            # Call load
            model, tokenizer = ModelQuantizer.load(self.temp_dir)
            
            # Check that the tokenizer was loaded
            mock_tokenizer.from_pretrained.assert_called_once_with(self.temp_dir)
            
            # Check that the model was loaded with the correct parameters
            mock_model.from_pretrained.assert_called_once()
            call_args = mock_model.from_pretrained.call_args[0]
            call_kwargs = mock_model.from_pretrained.call_args[1]
            self.assertEqual(call_args[0], self.temp_dir)
            self.assertEqual(call_kwargs["device_map"], "cpu")
            
            # Check that the returned model and tokenizer are correct
            self.assertEqual(model, mock_model_instance)
            self.assertEqual(tokenizer, mock_tokenizer_instance)


if __name__ == "__main__":
    unittest.main() 