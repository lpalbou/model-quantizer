#!/usr/bin/env python3
"""
Tests for the utils module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
import json

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantizer.utils import (
    get_model_size,
    calculate_memory_usage,
    publish_to_hub,
    create_model_card,
    setup_logging
)


class TestUtils(unittest.TestCase):
    """Tests for the utils module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch("os.path.getsize")
    @patch("os.walk")
    def test_get_model_size(self, mock_walk, mock_getsize):
        """Test that get_model_size returns the correct size."""
        # Mock os.walk to return a list of files
        mock_walk.return_value = [
            (self.temp_dir, [], ["file1.bin", "file2.bin"]),
            (os.path.join(self.temp_dir, "subdir"), [], ["file3.bin"])
        ]
        
        # Mock os.path.getsize to return file sizes
        def getsize_side_effect(path):
            if path.endswith("file1.bin"):
                return 1024  # 1 KB
            elif path.endswith("file2.bin"):
                return 2048  # 2 KB
            elif path.endswith("file3.bin"):
                return 4096  # 4 KB
            return 0
        
        mock_getsize.side_effect = getsize_side_effect
        
        # Call get_model_size
        size_bytes = get_model_size(self.temp_dir)
        
        # Check that the size is correct (1KB + 2KB + 4KB = 7KB = 7168 bytes)
        self.assertEqual(size_bytes, 7168)
    
    def test_calculate_memory_usage(self):
        """Test that calculate_memory_usage returns the correct memory usage."""
        # Test with different model sizes and bit widths
        
        # 1 GB model at 32-bit (original)
        original_size_gb = 1.0
        original_size_bytes = original_size_gb * 1024 * 1024 * 1024
        
        # 4-bit quantization should be approximately 1/8 of the original size
        memory_usage_4bit = calculate_memory_usage(original_size_bytes, 4)
        self.assertAlmostEqual(memory_usage_4bit / (1024 * 1024 * 1024), 0.125, places=2)
        
        # 8-bit quantization should be approximately 1/4 of the original size
        memory_usage_8bit = calculate_memory_usage(original_size_bytes, 8)
        self.assertAlmostEqual(memory_usage_8bit / (1024 * 1024 * 1024), 0.25, places=2)
    
    @patch("quantizer.utils.create_repo")
    @patch("quantizer.utils.create_model_card")
    def test_publish_to_hub(self, mock_create_model_card, mock_create_repo):
        """Test that publish_to_hub calls the correct functions."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock config
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"bits": 4, "method": "gptq"}
        mock_config.model_name = "test/model"
        
        # Call publish_to_hub
        repo_id = "username/model"
        url = publish_to_hub(mock_model, mock_tokenizer, mock_config, repo_id)
        
        # Check that create_repo was called
        mock_create_repo.assert_called_once_with(repo_id, private=False, exist_ok=True)
        
        # Check that create_model_card was called
        mock_create_model_card.assert_called_once_with(repo_id, mock_config)
        
        # Check that the model and tokenizer were pushed
        mock_model.push_to_hub.assert_called_once_with(repo_id, commit_message="Upload quantized model")
        mock_tokenizer.push_to_hub.assert_called_once_with(repo_id, commit_message="Upload quantized model")
        
        # Check that the returned URL is correct
        self.assertEqual(url, f"https://huggingface.co/{repo_id}")
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("quantizer.utils.upload_file")
    def test_create_model_card(self, mock_upload_file, mock_file):
        """Test that create_model_card creates a model card with the correct content."""
        # Mock config
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            "bits": 4,
            "method": "gptq",
            "group_size": 128,
            "desc_act": True,
            "sym": False
        }
        mock_config.model_name = "test/model"
        
        # Call create_model_card
        repo_id = "username/model"
        create_model_card(repo_id, mock_config)
        
        # Check that open was called to write the model card
        mock_file.assert_called_once()
        
        # Check that the model card contains the expected content
        written_data = mock_file().write.call_args[0][0]
        self.assertIn("# Quantized test/model", written_data)
        self.assertIn("This model is a quantized version of [test/model]", written_data)
        self.assertIn("- **Bits**: 4", written_data)
        self.assertIn("- **Method**: gptq", written_data)
        self.assertIn("- **Group Size**: 128", written_data)
        self.assertIn("- **Desc Act**: True", written_data)
        self.assertIn("- **Sym**: False", written_data)
        
        # Check that upload_file was called
        mock_upload_file.assert_called_once()
        self.assertEqual(mock_upload_file.call_args[0][1], repo_id)
        self.assertEqual(mock_upload_file.call_args[0][2], "README.md")
    
    @patch("logging.basicConfig")
    def test_setup_logging(self, mock_basic_config):
        """Test that setup_logging configures logging correctly."""
        # Call setup_logging with default parameters
        setup_logging()
        
        # Check that basicConfig was called with the correct parameters
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        self.assertEqual(call_kwargs["level"], 20)  # INFO level
        self.assertEqual(call_kwargs["format"], "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Reset mock and call with custom level
        mock_basic_config.reset_mock()
        setup_logging(level=10)  # DEBUG level
        
        # Check that basicConfig was called with the correct parameters
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        self.assertEqual(call_kwargs["level"], 10)  # DEBUG level


if __name__ == "__main__":
    unittest.main() 