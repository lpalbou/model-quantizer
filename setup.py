#!/usr/bin/env python3
"""
Setup script for the quantizer package.
"""

from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="model-quantizer",
    version="0.1.0",
    author="Laurent-Philippe Albou",
    author_email="laurent.albou@gmail.com",
    description="A tool for quantizing and saving Hugging Face models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lpalbou/model-quantizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "optimum>=1.12.0",
        "psutil>=5.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "accelerate>=0.20.0",
        "datasets>=2.10.0",
        "huggingface_hub>=0.16.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "jinja2>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "awq": ["autoawq>=0.1.0"],
        "bitsandbytes": ["bitsandbytes>=0.40.0"],
    },
    entry_points={
        "console_scripts": [
            "model-quantizer=quantizer.cli:main",
            "benchmark-model=benchmark_model:main",
            "visualize-benchmark=visualize_benchmark:main",
            "chat-with-model=chat_with_model:main",
            "run-benchmark=run_benchmark:main_cli",
        ],
    },
) 