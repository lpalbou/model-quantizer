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
            "std_memory": np.std(self.memory_samples),
            "current_memory": self.get_current_memory(),
            "initial_memory": self.initial_memory,
            "memory_increase": self.get_current_memory() - self.initial_memory
        }

class PerformanceTracker:
    """Track performance metrics during model interaction."""
    def __init__(self):
        self.start_time = time.time()
        self.load_duration = 0
        self.prompt_eval_count = 0
        self.prompt_eval_duration = 0
        self.eval_count = 0
        self.eval_duration = 0
        self.generation_samples = []
    
    def set_load_duration(self, duration):
        """Set the model loading duration."""
        self.load_duration = duration
    
    def add_prompt_eval(self, token_count, duration):
        """Add prompt evaluation metrics."""
        self.prompt_eval_count += token_count
        self.prompt_eval_duration += duration
    
    def add_generation(self, token_count, duration):
        """Add generation metrics."""
        self.eval_count += token_count
        self.eval_duration += duration
        if token_count > 0 and duration > 0:
            self.generation_samples.append(token_count / duration)
    
    def get_stats(self):
        """Get performance statistics."""
        total_duration = time.time() - self.start_time
        prompt_eval_rate = self.prompt_eval_count / self.prompt_eval_duration if self.prompt_eval_duration > 0 else 0
        eval_rate = self.eval_count / self.eval_duration if self.eval_duration > 0 else 0
        
        stats = {
            "total_duration": total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": self.prompt_eval_duration,
            "prompt_eval_rate": prompt_eval_rate,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
            "eval_rate": eval_rate,
        }
        
        if self.generation_samples:
            stats.update({
                "min_generation_rate": min(self.generation_samples),
                "max_generation_rate": max(self.generation_samples),
                "avg_generation_rate": np.mean(self.generation_samples),
                "std_generation_rate": np.std(self.generation_samples)
            })
        
        return stats

class ChatSession:
    """Manage a chat session with a model."""
    def __init__(self, model_path, device="cpu", max_new_tokens=256, system_prompt=None):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        
        # Initialize trackers
        self.memory_tracker = MemoryTracker()
        self.perf_tracker = PerformanceTracker()
        
        # Chat history
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        
        # Load model
        self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with performance tracking."""
        print(f"Loading model from {self.model_path}...")
        print(f"Initial memory usage: {self.memory_tracker.get_current_memory():.2f} GB")
        
        # Set environment variables for memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Measure loading time
        start_time = time.time()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        
        load_time = time.time() - start_time
        self.perf_tracker.set_load_duration(load_time)
        
        current_memory = self.memory_tracker.update()
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Current memory usage: {current_memory:.2f} GB")
        
        return model, tokenizer
    
    def _format_prompt(self):
        """Format the chat history into a prompt for the model."""
        formatted_prompt = ""
        for message in self.messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        # Add the assistant prefix for the next response
        if not formatted_prompt.endswith("Assistant: "):
            formatted_prompt += "Assistant: "
        
        return formatted_prompt
    
    def add_message(self, role, content):
        """Add a message to the chat history."""
        self.messages.append({"role": role, "content": content})
    
    def generate_response(self):
        """Generate a response based on the chat history."""
        # Update memory stats
        self.memory_tracker.update()
        
        # Format prompt
        prompt = self._format_prompt()
        
        # Tokenize prompt
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        prompt_tokens = len(input_ids[0])
        tokenize_time = time.time() - start_time
        self.perf_tracker.add_prompt_eval(prompt_tokens, tokenize_time)
        
        # Set up streamer for token-by-token generation
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        # Start generation
        start_time = time.time()
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Collect and print generated text
        generated_text = ""
        print("\nAssistant: ", end="", flush=True)
        for text in streamer:
            generated_text += text
            print(text, end="", flush=True)
        print("\n")
        
        # Update performance stats
        generation_time = time.time() - start_time
        generated_tokens = len(self.tokenizer.encode(generated_text)) - prompt_tokens
        self.perf_tracker.add_generation(generated_tokens, generation_time)
        
        # Update memory stats
        self.memory_tracker.update()
        
        # Extract just the assistant's response (not the whole prompt)
        response = generated_text.strip()
        
        # Add the response to the chat history
        self.add_message("assistant", response)
        
        return response, generation_time, generated_tokens
    
    def save_chat_history(self, filename=None):
        """Save the chat history to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.messages, f, indent=2)
        
        print(f"Chat history saved to {filename}")
        return filename
    
    def load_chat_history(self, filename):
        """Load chat history from a file."""
        with open(filename, 'r') as f:
            self.messages = json.load(f)
        
        print(f"Chat history loaded from {filename}")
    
    def print_stats(self):
        """Print performance and memory statistics."""
        memory_stats = self.memory_tracker.get_stats()
        perf_stats = self.perf_tracker.get_stats()
        
        print("\n=== Memory Usage ===")
        print(f"Initial memory:   {memory_stats['initial_memory']:.2f} GB")
        print(f"Current memory:   {memory_stats['current_memory']:.2f} GB")
        print(f"Memory increase:  {memory_stats['memory_increase']:.2f} GB")
        print(f"Min memory:       {memory_stats['min_memory']:.2f} GB")
        print(f"Max memory:       {memory_stats['max_memory']:.2f} GB")
        print(f"Avg memory:       {memory_stats['avg_memory']:.2f} GB")
        print(f"Std dev memory:   {memory_stats['std_memory']:.2f} GB")
        
        print("\n=== Performance Metrics ===")
        print(f"Total duration:       {perf_stats['total_duration']:.2f} seconds")
        print(f"Load duration:        {perf_stats['load_duration']:.2f} seconds")
        print(f"Prompt eval count:    {perf_stats['prompt_eval_count']} tokens")
        print(f"Prompt eval duration: {perf_stats['prompt_eval_duration']:.2f} seconds")
        print(f"Prompt eval rate:     {perf_stats['prompt_eval_rate']:.2f} tokens/second")
        print(f"Eval count:           {perf_stats['eval_count']} tokens")
        print(f"Eval duration:        {perf_stats['eval_duration']:.2f} seconds")
        print(f"Eval rate:            {perf_stats['eval_rate']:.2f} tokens/second")
        
        if "avg_generation_rate" in perf_stats:
            print("\n=== Generation Rate Statistics ===")
            print(f"Min generation rate: {perf_stats['min_generation_rate']:.2f} tokens/second")
            print(f"Max generation rate: {perf_stats['max_generation_rate']:.2f} tokens/second")
            print(f"Avg generation rate: {perf_stats['avg_generation_rate']:.2f} tokens/second")
            print(f"Std dev gen rate:    {perf_stats['std_generation_rate']:.2f} tokens/second")

def main():
    parser = argparse.ArgumentParser(description="Chat with a quantized model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the quantized model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on (cpu, cuda, mps)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--system_prompt", type=str, help="System prompt to use")
    parser.add_argument("--load_history", type=str, help="Load chat history from file")
    args = parser.parse_args()
    
    # Initialize chat session
    session = ChatSession(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt
    )
    
    # Load chat history if specified
    if args.load_history:
        session.load_chat_history(args.load_history)
    
    # Print welcome message
    print("\nChat with the model (type 'exit' to quit, 'stats' for metrics, 'save' to save chat history)")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Max tokens: {args.max_new_tokens}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() == "exit":
            break
        
        # Check for stats command
        if user_input.lower() == "stats":
            session.print_stats()
            continue
        
        # Check for save command
        if user_input.lower() == "save":
            session.save_chat_history()
            continue
        
        # Add user message to chat history
        session.add_message("user", user_input)
        
        # Generate response
        response, generation_time, generated_tokens = session.generate_response()
        
        # Print generation stats
        print(f"[Generated {generated_tokens} tokens in {generation_time:.2f}s ({generated_tokens/generation_time:.2f} tokens/s)]")
    
    # Print final stats
    print("\nFinal Statistics:")
    session.print_stats()

if __name__ == "__main__":
    main() 