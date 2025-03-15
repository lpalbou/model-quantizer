#!/usr/bin/env python3
"""
Integration tests for the quantizer module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import json

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantizer.model_quantizer import ModelQuantizer
from quantizer.quantization_config import QuantizationConfig
from quantizer.cli import main as cli_main


class TestIntegration(unittest.TestCase):
    """Integration tests for the quantizer module."""
    
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
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_end_to_end_quantization(self, mock_tokenizer, mock_model):
        """Test the end-to-end quantization process."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create quantizer
        quantizer = ModelQuantizer(self.config)
        
        # Quantize the model
        model, tokenizer = quantizer.quantize(self.model_name)
        
        # Check that the model and tokenizer were loaded
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        
        # Save the model
        quantizer.save()
        
        # Check that the model and tokenizer were saved
        mock_model_instance.save_pretrained.assert_called_once_with(self.temp_dir)
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(self.temp_dir)
        
        # Check that the config was saved
        config_path = os.path.join(self.temp_dir, "quantization_config.json")
        self.assertTrue(os.path.exists(config_path))
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("huggingface_hub.create_repo")
    @patch("huggingface_hub.upload_file")
    def test_end_to_end_with_publishing(self, mock_upload_file, mock_create_repo, mock_tokenizer, mock_model):
        """Test the end-to-end quantization process with publishing."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Create quantizer
        quantizer = ModelQuantizer(self.config)
        
        # Quantize the model
        model, tokenizer = quantizer.quantize(self.model_name)
        
        # Save the model
        quantizer.save()
        
        # Publish the model
        repo_id = "username/model"
        url = quantizer.publish_to_hub(repo_id)
        
        # Check that the repository was created
        mock_create_repo.assert_called_once_with(repo_id, private=False, exist_ok=True)
        
        # Check that the model and tokenizer were pushed
        mock_model_instance.push_to_hub.assert_called_once_with(repo_id, commit_message="Upload quantized model")
        mock_tokenizer_instance.push_to_hub.assert_called_once_with(repo_id, commit_message="Upload quantized model")
        
        # Check that the model card was created
        mock_upload_file.assert_called_once()
        self.assertEqual(mock_upload_file.call_args[0][1], repo_id)
        self.assertEqual(mock_upload_file.call_args[0][2], "README.md")
    
    @patch("sys.argv")
    @patch("quantizer.cli.ModelQuantizer")
    def test_cli_integration(self, mock_quantizer, mock_argv):
        """Test the CLI integration."""
        # Mock the ModelQuantizer instance
        mock_instance = MagicMock()
        mock_quantizer.return_value = mock_instance
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_instance.quantize.return_value = (mock_model, mock_tokenizer)
        
        # Set up CLI arguments
        mock_argv.__getitem__.side_effect = lambda idx: [
            "model-quantizer",
            self.model_name,
            "--bits", "4",
            "--method", "gptq",
            "--output-dir", self.temp_dir,
            "--group-size", "64",
            "--desc-act",
            "--device", "cpu"
        ][idx]
        
        # Run the CLI
        cli_main()
        
        # Check that ModelQuantizer was created with the correct config
        mock_quantizer.assert_called_once()
        config = mock_quantizer.call_args[0][0]
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.method, "gptq")
        self.assertEqual(config.output_dir, self.temp_dir)
        self.assertEqual(config.group_size, 64)
        self.assertTrue(config.desc_act)
        self.assertEqual(config.device, "cpu")
        
        # Check that quantize was called with the correct arguments
        mock_instance.quantize.assert_called_once_with(self.model_name)
        
        # Check that save was called
        mock_instance.save.assert_called_once()


if __name__ == "__main__":
    unittest.main() 