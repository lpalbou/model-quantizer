#!/usr/bin/env python3
"""
Comprehensive chat script for interacting with quantized models.
Features:
- Detailed memory and performance tracking
- Chat history management
- Token-by-token streaming
- System prompt customization
- Response formatting
"""

import os
import time
import json
import argparse
import psutil
import torch
import numpy as np
from datetime import datetime
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class MemoryTracker:
    """Track memory usage during model interaction."""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.min_memory = float('inf')
        self.max_memory = 0
        self.memory_samples = []
        self.initial_memory = self.get_current_memory()
        self.memory_samples.append(self.initial_memory)
    
    def get_current_memory(self):
        """Get current memory usage in GB."""
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024 ** 3)  # Convert to GB
    
    def update(self):
        """Update memory statistics."""
        current = self.get_current_memory()
        self.min_memory = min(self.min_memory, current)
        self.max_memory = max(self.max_memory, current)
        self.memory_samples.append(current)
        return current
    
    def get_stats(self):
        """Get memory statistics."""
        return {
            "min_memory": self.min_memory,
            "max_memory": self.max_memory,
            "avg_memory": np.mean(self.memory_samples),
            "initial_memory": self.initial_memory,
            "memory_increase": self.max_memory - self.initial_memory,
            "current_memory": self.memory_samples[-1]
        }

class PerformanceTracker:
    """Track performance metrics during model interaction."""
    def __init__(self):
        self.load_duration = 0
        self.prompt_tokens = 0
        self.prompt_eval_time = 0
        self.generated_tokens = 0
        self.generation_time = 0
    
    def set_load_duration(self, duration):
        """Set model loading duration."""
        self.load_duration = duration
    
    def add_prompt_eval(self, token_count, duration):
        """Add prompt evaluation metrics."""
        self.prompt_tokens += token_count
        self.prompt_eval_time += duration
    
    def add_generation(self, token_count, duration):
        """Add generation metrics."""
        self.generated_tokens += token_count
        self.generation_time += duration
    
    def get_stats(self):
        """Get performance statistics."""
        prompt_tokens_per_sec = self.prompt_tokens / self.prompt_eval_time if self.prompt_eval_time > 0 else 0
        generation_tokens_per_sec = self.generated_tokens / self.generation_time if self.generation_time > 0 else 0
        
        return {
            "load_time": f"{self.load_duration:.2f}s",
            "prompt_tokens": self.prompt_tokens,
            "prompt_eval_time": f"{self.prompt_eval_time:.2f}s",
            "prompt_tokens_per_sec": f"{prompt_tokens_per_sec:.2f}",
            "generated_tokens": self.generated_tokens,
            "generation_time": f"{self.generation_time:.2f}s",
            "generation_tokens_per_sec": f"{generation_tokens_per_sec:.2f}",
            "total_tokens": self.prompt_tokens + self.generated_tokens,
            "total_time": f"{self.load_duration + self.prompt_eval_time + self.generation_time:.2f}s"
        }

class ChatSession:
    """Interactive chat session with a model."""
    def __init__(self, model_path, device="cpu", max_new_tokens=256, system_prompt=None):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Set default system prompt if none provided
        if system_prompt is None:
            self.system_prompt = "You are a helpful AI assistant."
        else:
            self.system_prompt = system_prompt
        
        self.memory_tracker = MemoryTracker()
        self.performance_tracker = PerformanceTracker()
        self.chat_history = []
        self.add_message("system", self.system_prompt)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Initial memory usage: {self.memory_tracker.get_current_memory():.2f} GB")
        print(f"Loading model from {self.model_path}...")
        
        start_time = time.time()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        load_duration = time.time() - start_time
        self.performance_tracker.set_load_duration(load_duration)
        
        current_memory = self.memory_tracker.update()
        print(f"Model loaded in {load_duration:.2f} seconds")
        print(f"Current memory usage: {current_memory:.2f} GB")
        print(f"Memory increase: {current_memory - self.memory_tracker.initial_memory:.2f} GB")
        
        return model, tokenizer
    
    def _format_prompt(self):
        """Format the chat history into a prompt for the model."""
        formatted_prompt = ""
        
        for message in self.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                # System message is typically not shown in the prompt
                continue
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        # Add the assistant prefix for the next response
        formatted_prompt += "Assistant: "
        
        return formatted_prompt
    
    def add_message(self, role, content):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    
    def generate_response(self):
        """Generate a response from the model."""
        prompt = self._format_prompt()
        
        # Tokenize the prompt
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_count = inputs.input_ids.shape[1]
        tokenize_time = time.time() - start_time
        
        # Set up the streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Start generation in a separate thread
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "streamer": streamer
        }
        
        # Update memory and performance trackers
        self.memory_tracker.update()
        self.performance_tracker.add_prompt_eval(input_token_count, tokenize_time)
        
        # Start generation
        start_time = time.time()
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        generated_text = ""
        for text in streamer:
            generated_text += text
            print(text, end="", flush=True)
        
        print()  # Add a newline after generation
        
        # Calculate generation time and token count
        generation_time = time.time() - start_time
        generated_token_count = len(self.tokenizer.encode(generated_text))
        
        # Update trackers
        self.memory_tracker.update()
        self.performance_tracker.add_generation(generated_token_count, generation_time)
        
        # Add the response to chat history
        self.add_message("assistant", generated_text)
        
        return generated_text
    
    def save_chat_history(self, filename=None):
        """Save chat history to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(self.chat_history, f, indent=2)
        
        print(f"Chat history saved to {filename}")
        return filename
    
    def load_chat_history(self, filename):
        """Load chat history from a file."""
        with open(filename, "r") as f:
            self.chat_history = json.load(f)
        
        print(f"Chat history loaded from {filename}")
    
    def print_stats(self):
        """Print performance and memory statistics."""
        memory_stats = self.memory_tracker.get_stats()
        performance_stats = self.performance_tracker.get_stats()
        
        print("\n" + "="*50)
        print("CHAT SESSION STATISTICS")
        print("="*50)
        
        print("\nMemory Usage:")
        print(f"  Initial: {memory_stats['initial_memory']:.2f} GB")
        print(f"  Current: {memory_stats['current_memory']:.2f} GB")
        print(f"  Min: {memory_stats['min_memory']:.2f} GB")
        print(f"  Max: {memory_stats['max_memory']:.2f} GB")
        print(f"  Avg: {memory_stats['avg_memory']:.2f} GB")
        print(f"  Increase: {memory_stats['memory_increase']:.2f} GB")
        
        print("\nPerformance:")
        print(f"  Model Load Time: {performance_stats['load_time']}")
        print(f"  Prompt Tokens: {performance_stats['prompt_tokens']}")
        print(f"  Prompt Evaluation Time: {performance_stats['prompt_eval_time']}")
        print(f"  Prompt Processing Speed: {performance_stats['prompt_tokens_per_sec']} tokens/sec")
        print(f"  Generated Tokens: {performance_stats['generated_tokens']}")
        print(f"  Generation Time: {performance_stats['generation_time']}")
        print(f"  Generation Speed: {performance_stats['generation_tokens_per_sec']} tokens/sec")
        print(f"  Total Tokens: {performance_stats['total_tokens']}")
        print(f"  Total Time: {performance_stats['total_time']}")
        
        print("="*50)

def main():
    """Main function for the chat script."""
    parser = argparse.ArgumentParser(description="Chat with a quantized model")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (cpu, cuda, mps)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--system_prompt", help="Custom system prompt")
    parser.add_argument("--load_history", help="Load chat history from file")
    args = parser.parse_args()
    
    # Create chat session
    chat = ChatSession(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt
    )
    
    # Load chat history if specified
    if args.load_history:
        chat.load_chat_history(args.load_history)
    
    print(f"\nChat with {args.model_path}")
    print("Type 'exit' to end the conversation")
    print("Type 'save' to save the chat history")
    print("Type 'stats' to show performance statistics")
    print("="*50)
    
    # Main chat loop
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "save":
            chat.save_chat_history()
            continue
        elif user_input.lower() == "stats":
            chat.print_stats()
            continue
        
        # Add user message and generate response
        chat.add_message("user", user_input)
        print("\nAssistant: ", end="", flush=True)
        chat.generate_response()
    
    # Print final stats
    chat.print_stats()

if __name__ == "__main__":
    main() 