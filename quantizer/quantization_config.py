"""
Configuration class for model quantization.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.
    
    Attributes:
        bits (int): Number of bits for quantization (2, 3, 4, or 8).
            Note: 4-bit and 8-bit are recommended for GPTQ.
        method (str): Quantization method to use ('gptq', 'awq', or 'bitsandbytes').
        output_dir (str): Directory to save the quantized model.
        calibration_dataset (Union[List[str], str, None]): Dataset for calibration.
            Can be a list of strings, a dataset name, or None.
        group_size (int): Group size for quantization (default: 128).
        desc_act (bool): Whether to use descending activation order (default: False).
        sym (bool): Whether to use symmetric quantization (default: True).
        device (str): Device to use for quantization ('cpu', 'cuda', 'mps').
        use_optimum (bool): Whether to use the optimum library for quantization (default: True).
        additional_params (Dict[str, Any]): Additional parameters for specific quantization methods.
    """
    
    bits: int = 8
    method: str = "gptq"
    output_dir: str = "quantized-model"
    calibration_dataset: Optional[Union[List[str], str]] = None
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    device: str = "auto"
    use_optimum: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate bits
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Bits must be one of [2, 3, 4, 8], got {self.bits}")
        
        # Add warnings for non-recommended bit widths
        if self.method == "gptq" and self.bits in [2, 3]:
            logger.warning(f"{self.bits}-bit quantization with GPTQ is experimental and may not be fully supported.")
            logger.warning("4-bit or 8-bit quantization is recommended for GPTQ.")
        
        if self.method == "bitsandbytes" and self.bits not in [4, 8]:
            raise ValueError(f"BitsAndBytes only supports 4 or 8 bits, got {self.bits}")
        
        if self.method == "awq" and self.bits != 4:
            logger.warning(f"AWQ is primarily designed for 4-bit quantization, got {self.bits}-bit.")
            logger.warning("Results with other bit widths may not be optimal.")
        
        # Validate method
        valid_methods = ["gptq", "awq", "bitsandbytes"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {self.method}")
        
        # Set default calibration dataset if None
        if self.calibration_dataset is None:
            self.calibration_dataset = [
                "This is a sample text for calibration.",
                "The model will be quantized using this text.",
                "Quantization reduces the model size while maintaining performance."
            ]
        
        # Validate device
        if self.device != "auto" and self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Device must be one of ['auto', 'cpu', 'cuda', 'mps'], got {self.device}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "bits": self.bits,
            "method": self.method,
            "output_dir": self.output_dir,
            "calibration_dataset": self.calibration_dataset,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "device": self.device,
            "use_optimum": self.use_optimum,
            "additional_params": self.additional_params
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create a configuration from a dictionary."""
        return cls(**config_dict) 