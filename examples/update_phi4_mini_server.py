#!/usr/bin/env python3
"""
Script to update the Phi4MiniServer to use the quantized model.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def update_server_file(server_file_path, quantized_model_path):
    """
    Update the Phi4MiniServer to use the quantized model.
    
    Args:
        server_file_path: Path to the Phi4MiniServer file.
        quantized_model_path: Path to the quantized model.
    """
    # Read the server file
    with open(server_file_path, "r") as f:
        content = f.read()
    
    # Update the DEFAULT_MODEL constant
    content = content.replace(
        'DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"',
        f'DEFAULT_MODEL = "{quantized_model_path}"'
    )
    
    # Update the _load_model method to use the quantized model
    load_model_start = content.find("def _load_model(self):")
    if load_model_start == -1:
        logger.error("Could not find _load_model method in server file")
        return False
    
    # Find the end of the method
    load_model_end = content.find("def ", load_model_start + 1)
    if load_model_end == -1:
        load_model_end = len(content)
    
    # Extract the method
    load_model_method = content[load_model_start:load_model_end]
    
    # Create the new method
    new_load_model_method = """
    def _load_model(self):
        \"\"\"Load the quantized model and tokenizer.\"\"\"
        try:
            logger.info(f"Loading model {self.model_name}...")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            tokenizer_time = time.time() - tokenizer_start
            logger.info(f"Tokenizer loaded in {tokenizer_time:.2f} seconds")
            
            # Log initial memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024 ** 3)
            logger.info(f"Initial process memory usage: {memory_usage_gb:.2f} GB")
            
            # Load model
            logger.info("Loading quantized model...")
            model_start = time.time()
            
            # Load the quantized model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            
            model_time = time.time() - model_start
            logger.info(f"Model loaded in {model_time:.2f} seconds")
            
            # Log memory usage after loading
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024 ** 3)
            logger.info(f"Process memory usage after loading: {memory_usage_gb:.2f} GB")
            
            # Get detailed memory info
            memory_full = process.memory_full_info()
            logger.info(f"Memory details:")
            logger.info(f"  - RSS (Resident Set Size): {memory_full.rss / (1024**3):.2f} GB")
            logger.info(f"  - VMS (Virtual Memory Size): {memory_full.vms / (1024**3):.2f} GB")
            if hasattr(memory_full, 'uss'):
                logger.info(f"  - USS (Unique Set Size): {memory_full.uss / (1024**3):.2f} GB")
            if hasattr(memory_full, 'pss'):
                logger.info(f"  - PSS (Proportional Set Size): {memory_full.pss / (1024**3):.2f} GB")
            
            # Log model size
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e9  # billions
            logger.info(f"Model size: {model_size:.2f}B parameters")
            
            # Get model device
            model_device = next(self.model.parameters()).device
            logger.info(f"Model loaded on device: {model_device}")
            
            # Optimize memory usage
            self._optimize_memory()
            
            # Signal that the model is loaded
            self._model_loaded_event.set()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Ensure the event is set to avoid deadlocks
            self._model_loaded_event.set()
    """
    
    # Replace the method
    content = content.replace(load_model_method, new_load_model_method)
    
    # Remove the bits parameter from the __init__ method
    init_start = content.find("def __init__(self,")
    if init_start == -1:
        logger.error("Could not find __init__ method in server file")
        return False
    
    # Find the end of the method signature
    init_end = content.find(":", init_start)
    if init_end == -1:
        logger.error("Could not find end of __init__ method signature in server file")
        return False
    
    # Extract the method signature
    init_signature = content[init_start:init_end]
    
    # Create the new method signature without the bits parameter
    new_init_signature = init_signature.replace(", bits: int = 4", "")
    
    # Replace the method signature
    content = content.replace(init_signature, new_init_signature)
    
    # Remove the bits parameter from the initialization
    init_body_start = content.find("self.bits = bits", init_end)
    if init_body_start != -1:
        init_body_end = content.find("\n", init_body_start)
        if init_body_end != -1:
            content = content[:init_body_start] + content[init_body_end:]
    
    # Update the logger.info line in __init__
    log_line_start = content.find('logger.info(f"Initialized Phi-4-Mini Server with model:', init_end)
    if log_line_start != -1:
        log_line_end = content.find("\n", log_line_start)
        if log_line_end != -1:
            old_log_line = content[log_line_start:log_line_end]
            new_log_line = old_log_line.replace(", bits: {bits}", "")
            content = content.replace(old_log_line, new_log_line)
    
    # Update the main block to remove the bits parameter
    main_block_start = content.find("if __name__ == \"__main__\":")
    if main_block_start != -1:
        # Find the parser.add_argument line for bits
        bits_arg_start = content.find('parser.add_argument("--bits"', main_block_start)
        if bits_arg_start != -1:
            bits_arg_end = content.find(")\n", bits_arg_start)
            if bits_arg_end != -1:
                content = content[:bits_arg_start] + content[bits_arg_end + 2:]
        
        # Update the server creation line
        server_creation_start = content.find("server = Phi4MiniServer(", main_block_start)
        if server_creation_start != -1:
            server_creation_end = content.find(")", server_creation_start)
            if server_creation_end != -1:
                old_server_creation = content[server_creation_start:server_creation_end + 1]
                new_server_creation = old_server_creation.replace(", bits=args.bits", "")
                content = content.replace(old_server_creation, new_server_creation)
        
        # Update the print line
        print_line_start = content.find('print(f"Starting Phi-4-Mini server with {args.bits}-bit', main_block_start)
        if print_line_start != -1:
            print_line_end = content.find("\n", print_line_start)
            if print_line_end != -1:
                old_print_line = content[print_line_start:print_line_end]
                new_print_line = 'print(f"Starting Phi-4-Mini server with quantized model...")'
                content = content.replace(old_print_line, new_print_line)
    
    # Write the updated content back to the file
    with open(server_file_path, "w") as f:
        f.write(content)
    
    logger.info(f"Updated server file: {server_file_path}")
    return True

def main():
    """Main entry point."""
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: update_phi4_mini_server.py <server_file_path> <quantized_model_path>")
        return 1
    
    server_file_path = sys.argv[1]
    quantized_model_path = sys.argv[2]
    
    # Check if the server file exists
    if not os.path.exists(server_file_path):
        logger.error(f"Server file not found: {server_file_path}")
        return 1
    
    # Check if the quantized model exists
    if not os.path.exists(quantized_model_path):
        logger.error(f"Quantized model not found: {quantized_model_path}")
        return 1
    
    # Update the server file
    if not update_server_file(server_file_path, quantized_model_path):
        logger.error("Failed to update server file")
        return 1
    
    logger.info("Server file updated successfully")
    
    # Print usage instructions
    print("\nTo use the updated server:")
    print("1. Start the server:")
    print(f"   python {server_file_path}")
    print("2. Send requests to the server:")
    print("   curl -X POST http://127.0.0.1:8000/generate -H \"Content-Type: application/json\" -d '{\"prompt\": \"What is the capital of France?\", \"system_prompt\": \"You are a helpful AI assistant.\", \"max_tokens\": 100, \"temperature\": 0.2}'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 