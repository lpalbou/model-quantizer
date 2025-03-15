"""
Model quantizer for Hugging Face models.
"""

import os
import sys
import json
import logging
import platform
import torch
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GPTQConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

from .quantization_config import QuantizationConfig
from .model_card import generate_model_card, update_model_card_with_benchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    A class for quantizing and saving Hugging Face models.
    
    This class supports different quantization methods:
    - GPTQ: Post-training quantization using the GPTQ algorithm
    - AWQ: Post-training quantization using the AWQ algorithm
    - BitsAndBytes: Quantization using the BitsAndBytes library
    
    The quantized model can be saved to disk for later use.
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize the model quantizer.
        
        Args:
            config: Configuration for quantization. If None, default configuration is used.
        """
        self.config = config or QuantizationConfig()
        self.model = None
        self.tokenizer = None
        
        # Check for required packages
        self._check_dependencies()
        
        # Check platform compatibility
        self._check_platform_compatibility()
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        if self.config.method == "gptq" and self.config.use_optimum:
            try:
                import optimum
                logger.info("Optimum package found.")
            except ImportError:
                logger.warning("Optimum package not found. Installing...")
                os.system("pip install optimum")
        
        if self.config.method == "bitsandbytes":
            try:
                import bitsandbytes
                logger.info("BitsAndBytes package found.")
            except ImportError:
                logger.warning("BitsAndBytes package not found. Installing...")
                os.system("pip install bitsandbytes")
        
        if self.config.method == "awq":
            try:
                import awq
                logger.info("AWQ package found.")
            except ImportError:
                logger.warning("AWQ package not found. Installing...")
                os.system("pip install autoawq")
    
    def _check_platform_compatibility(self):
        """Check if the selected quantization method is compatible with the current platform."""
        system = platform.system()
        
        if system == "Darwin" and self.config.method == "bitsandbytes":
            logger.warning("BitsAndBytes quantization is not fully supported on macOS.")
            logger.warning("Consider using GPTQ quantization instead for better compatibility.")
            
            # Ask for confirmation if in interactive mode
            if sys.stdin.isatty() and sys.stdout.isatty():
                response = input("Do you want to continue with BitsAndBytes anyway? (y/n): ")
                if response.lower() != "y":
                    logger.info("Switching to GPTQ quantization for better macOS compatibility.")
                    self.config.method = "gptq"
        
        if system == "Darwin" and self.config.device == "mps":
            # Set environment variable for MPS fallback
            if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
                logger.info("Setting PYTORCH_ENABLE_MPS_FALLBACK=1 for better MPS compatibility.")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    def _get_device(self) -> str:
        """
        Get the appropriate device for quantization.
        
        Returns:
            The device to use for quantization.
        """
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Set environment variable for MPS fallback
            if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
                logger.info("Setting PYTORCH_ENABLE_MPS_FALLBACK=1 for better MPS compatibility.")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            return "mps"
        else:
            return "cpu"
    
    def _prepare_gptq_config(self, tokenizer: PreTrainedTokenizer) -> GPTQConfig:
        """
        Prepare GPTQ configuration.
        
        Args:
            tokenizer: The tokenizer for the model.
            
        Returns:
            GPTQ configuration.
        """
        return GPTQConfig(
            bits=self.config.bits,
            dataset=self.config.calibration_dataset,
            tokenizer=tokenizer,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
            sym=self.config.sym,
            **self.config.additional_params
        )
    
    def _prepare_bnb_config(self) -> BitsAndBytesConfig:
        """
        Prepare BitsAndBytes configuration.
        
        Returns:
            BitsAndBytes configuration.
        """
        if self.config.bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                **self.config.additional_params
            )
        elif self.config.bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                **self.config.additional_params
            )
        else:
            raise ValueError(f"BitsAndBytes only supports 4 or 8 bits, got {self.config.bits}")
    
    def quantize(self, model_name: str, output_dir: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Quantize a model and save it to disk.
        
        Args:
            model_name: The name or path of the model to quantize.
            output_dir: Directory to save the quantized model. If None, use the one from config.
            
        Returns:
            The quantized model and tokenizer.
        """
        # Set output directory
        output_dir = output_dir or self.config.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get device
        device = self._get_device()
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Quantize model based on method
        if self.config.method == "gptq":
            self._quantize_gptq(model_name, device)
        elif self.config.method == "bitsandbytes":
            self._quantize_bnb(model_name, device)
        elif self.config.method == "awq":
            self._quantize_awq(model_name, device)
        else:
            raise ValueError(f"Unsupported quantization method: {self.config.method}")
        
        # Save model and tokenizer
        logger.info(f"Saving quantized model to: {output_dir}")
        self.save(output_dir)
        
        return self.model, self.tokenizer
    
    def _quantize_gptq(self, model_name: str, device: str):
        """
        Quantize a model using GPTQ.
        
        This method uses Hugging Face's implementation of the GPTQ algorithm.
        GPTQ is primarily designed for 4-bit and 8-bit quantization, though
        2-bit and 3-bit are supported experimentally.
        
        Args:
            model_name: The name or path of the model to quantize.
            device: The device to use for quantization.
        """
        logger.info(f"Quantizing model {model_name} with GPTQ ({self.config.bits}-bit)")
        
        # Prepare GPTQ configuration
        gptq_config = self._prepare_gptq_config(self.tokenizer)
        
        try:
            # Load and quantize model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else device,
                quantization_config=gptq_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            logger.info(f"Model quantized successfully with GPTQ ({self.config.bits}-bit)")
        except Exception as e:
            logger.error(f"Error quantizing model: {str(e)}")
            logger.error("If you encounter issues with GPTQ quantization, try:")
            logger.error("1. Using a different device (--device cpu or --device cuda)")
            logger.error("2. Using a different quantization method (--method bitsandbytes)")
            raise
    
    def _quantize_bnb(self, model_name: str, device: str):
        """
        Quantize a model using BitsAndBytes.
        
        This method uses the BitsAndBytes library which only supports
        4-bit and 8-bit quantization. BitsAndBytes is primarily designed
        for CUDA devices and may not work well on CPU or MPS.
        
        Args:
            model_name: The name or path of the model to quantize.
            device: The device to use for quantization.
        """
        logger.info(f"Quantizing model {model_name} with BitsAndBytes ({self.config.bits}-bit)")
        
        # Prepare BitsAndBytes configuration
        bnb_config = self._prepare_bnb_config()
        
        # Load and quantize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else device,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        logger.info(f"Model quantized successfully with BitsAndBytes ({self.config.bits}-bit)")
    
    def _quantize_awq(self, model_name: str, device: str):
        """
        Quantize a model using AWQ.
        
        This method uses the AWQ (Activation-aware Weight Quantization) algorithm.
        AWQ is primarily designed for 4-bit quantization and requires specific
        hardware support. Other bit widths may not provide optimal results.
        
        Args:
            model_name: The name or path of the model to quantize.
            device: The device to use for quantization.
        """
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            logger.error("AWQ package not found. Please install it with: pip install autoawq")
            raise
        
        logger.info(f"Quantizing model {model_name} with AWQ ({self.config.bits}-bit)")
        
        # Load and quantize model
        self.model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else device,
            bits=self.config.bits,
            trust_remote_code=True,
            **self.config.additional_params
        )
        
        logger.info(f"Model quantized successfully with AWQ ({self.config.bits}-bit)")
    
    def save(self, output_dir: Optional[str] = None, hf_username: Optional[str] = None):
        """
        Save the quantized model and tokenizer to disk.
        
        Args:
            output_dir: Directory to save the model. If None, use the one from config.
            hf_username: Hugging Face username for the model card. If None, a placeholder is used.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be quantized before saving.")
        
        # Set output directory
        output_dir = output_dir or self.config.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save quantization config
        config_path = os.path.join(output_dir, "quantization_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Generate model card
        model_name = os.path.basename(output_dir)
        original_model_name = self.config.model_name
        
        # Determine calibration dataset name
        calibration_dataset = "c4"
        if isinstance(self.config.calibration_dataset, str):
            calibration_dataset = self.config.calibration_dataset
        
        # Generate additional args string
        additional_args = ""
        if self.config.group_size != 128:
            additional_args += f" --group-size {self.config.group_size}"
        if self.config.desc_act:
            additional_args += " --desc-act"
        if not self.config.sym:
            additional_args += " --no-sym"
        
        logger.info(f"Generating model card for {model_name}")
        generate_model_card(
            model_name=model_name,
            original_model_name=original_model_name,
            method=self.config.method,
            bits=self.config.bits,
            output_dir=output_dir,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
            sym=self.config.sym,
            hf_username=hf_username,
            calibration_dataset=calibration_dataset,
            additional_args=additional_args
        )
        
        logger.info(f"Model saved to {output_dir}")
    
    def publish_to_hub(self, repo_id: str, private: bool = False, commit_message: str = "Upload quantized model"):
        """
        Publish the quantized model to the Hugging Face Hub.
        
        Args:
            repo_id: The repository ID to publish to (e.g., "username/model-name").
            private: Whether the repository should be private.
            commit_message: The commit message to use.
            
        Returns:
            The URL of the published model.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be quantized before publishing.")
        
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            logger.error("huggingface_hub package not found. Installing...")
            os.system("pip install huggingface_hub")
            from huggingface_hub import HfApi, create_repo
        
        logger.info(f"Publishing model to {repo_id}")
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id, private=private, exist_ok=True)
            logger.info(f"Repository {repo_id} created or already exists")
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise
        
        # Create model card if it doesn't exist
        model_card = self._create_model_card(repo_id)
        
        # Save model to temporary directory
        temp_dir = f"{self.config.output_dir}_temp"
        self.save(temp_dir)
        
        # Save model card
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(model_card)
        
        # Push to Hub
        try:
            self.model.push_to_hub(repo_id, commit_message=commit_message)
            self.tokenizer.push_to_hub(repo_id, commit_message=commit_message)
            logger.info(f"Model published to https://huggingface.co/{repo_id}")
            return f"https://huggingface.co/{repo_id}"
        except Exception as e:
            logger.error(f"Error publishing model: {e}")
            raise
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _create_model_card(self, repo_id: str) -> str:
        """
        Create a model card for the quantized model.
        
        Args:
            repo_id: The repository ID.
            
        Returns:
            The model card as a string.
        """
        # Extract original model name
        original_model = self.config.model_name if hasattr(self.config, "model_name") else "Unknown"
        
        # Extract username and model name from repo_id
        username, model_name = repo_id.split("/")
        
        # Determine calibration dataset name
        calibration_dataset = "c4"
        if isinstance(self.config.calibration_dataset, str):
            calibration_dataset = self.config.calibration_dataset
        
        # Generate additional args string
        additional_args = ""
        if self.config.group_size != 128:
            additional_args += f" --group-size {self.config.group_size}"
        if self.config.desc_act:
            additional_args += " --desc-act"
        if not self.config.sym:
            additional_args += " --no-sym"
        
        # Create a temporary directory to generate the model card
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Generate the model card
        model_card_path = generate_model_card(
            model_name=model_name,
            original_model_name=original_model,
            method=self.config.method,
            bits=self.config.bits,
            output_dir=temp_dir,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
            sym=self.config.sym,
            hf_username=username,
            calibration_dataset=calibration_dataset,
            additional_args=additional_args
        )
        
        # Read the model card
        with open(model_card_path, "r") as f:
            model_card = f.read()
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return model_card
    
    @classmethod
    def load(cls, model_dir: str, device: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a quantized model and tokenizer.
        
        Args:
            model_dir: Directory containing the quantized model.
            device: Device to load the model on. If None, use 'auto'.
            
        Returns:
            The loaded model and tokenizer.
        """
        # Load configuration if available
        config_path = os.path.join(model_dir, "quantization_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = QuantizationConfig.from_dict(config_dict)
        else:
            config = QuantizationConfig()
        
        # Override device if specified
        if device:
            config.device = device
        
        # Create instance
        instance = cls(config)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from: {model_dir}")
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        logger.info(f"Loading model from: {model_dir}")
        device_map = "auto" if config.device == "auto" or config.device == "cuda" else config.device
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        logger.info(f"Model and tokenizer loaded successfully from: {model_dir}")
        
        return instance.model, instance.tokenizer 