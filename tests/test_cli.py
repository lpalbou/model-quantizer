#!/usr/bin/env python3
"""
Tests for the CLI module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantizer.cli import parse_args, main


class TestCLI(unittest.TestCase):
    """Tests for the CLI module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_parse_args_default(self):
        """Test that parse_args returns default values."""
        args = parse_args(["model_name"])
        self.assertEqual(args.model_name, "model_name")
        self.assertEqual(args.bits, 8)
        self.assertEqual(args.method, "gptq")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.group_size, 128)
        self.assertFalse(args.desc_act)
        self.assertFalse(args.no_sym)
        self.assertFalse(args.no_optimum)
        self.assertIsNone(args.calibration_dataset)
        self.assertFalse(args.verbose)
        self.assertFalse(args.publish)
        self.assertIsNone(args.repo_id)
        self.assertFalse(args.private)
        self.assertEqual(args.commit_message, "Upload quantized model")
    
    def test_parse_args_custom(self):
        """Test that parse_args returns custom values."""
        args = parse_args([
            "model_name",
            "--bits", "4",
            "--method", "bitsandbytes",
            "--device", "cpu",
            "--group-size", "64",
            "--desc-act",
            "--no-sym",
            "--no-optimum",
            "--calibration-dataset", "test,dataset",
            "--verbose",
            "--publish",
            "--repo-id", "username/model",
            "--private",
            "--commit-message", "Custom message"
        ])
        self.assertEqual(args.model_name, "model_name")
        self.assertEqual(args.bits, 4)
        self.assertEqual(args.method, "bitsandbytes")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.group_size, 64)
        self.assertTrue(args.desc_act)
        self.assertTrue(args.no_sym)
        self.assertTrue(args.no_optimum)
        self.assertEqual(args.calibration_dataset, "test,dataset")
        self.assertTrue(args.verbose)
        self.assertTrue(args.publish)
        self.assertEqual(args.repo_id, "username/model")
        self.assertTrue(args.private)
        self.assertEqual(args.commit_message, "Custom message")
    
    @patch("quantizer.cli.ModelQuantizer")
    def test_main(self, mock_quantizer):
        """Test that main creates a ModelQuantizer and calls quantize."""
        # Mock the ModelQuantizer instance
        mock_instance = MagicMock()
        mock_quantizer.return_value = mock_instance
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_instance.quantize.return_value = (mock_model, mock_tokenizer)
        
        # Call main with minimal arguments
        with patch("sys.argv", ["model-quantizer", "model_name"]):
            main()
        
        # Check that ModelQuantizer was created with the correct config
        mock_quantizer.assert_called_once()
        config = mock_quantizer.call_args[0][0]
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.method, "gptq")
        
        # Check that quantize was called with the correct arguments
        mock_instance.quantize.assert_called_once_with("model_name")
    
    @patch("quantizer.cli.ModelQuantizer")
    def test_main_with_publish(self, mock_quantizer):
        """Test that main calls publish_to_hub when --publish is specified."""
        # Mock the ModelQuantizer instance
        mock_instance = MagicMock()
        mock_quantizer.return_value = mock_instance
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_instance.quantize.return_value = (mock_model, mock_tokenizer)
        
        # Mock the publish_to_hub method
        mock_instance.publish_to_hub.return_value = "https://huggingface.co/username/model"
        
        # Call main with publish arguments
        with patch("sys.argv", [
            "model-quantizer", "model_name",
            "--publish",
            "--repo-id", "username/model"
        ]):
            main()
        
        # Check that publish_to_hub was called with the correct arguments
        mock_instance.publish_to_hub.assert_called_once_with(
            repo_id="username/model",
            private=False,
            commit_message="Upload quantized model"
        )


if __name__ == "__main__":
    unittest.main() 