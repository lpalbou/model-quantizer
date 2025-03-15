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
    version="0.3.2",
    author="Laurent-Philippe Albou",
    author_email="laurent.albou@gmail.com",
    description="A tool for quantizing large language models",
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.30.0",
        "huggingface_hub>=0.16.0",
        "numpy==1.26.4",
        "psutil==7.0.0",
        "tqdm==4.67.1",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.0",
            "black==25.1.0",
            "isort==6.0.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "matplotlib==3.10.0",
            "colorama>=0.4.6",
            "jinja2==3.1.5",
        ],
        "data": [
            "datasets==3.4.0",
            "accelerate==1.5.2",
        ],
        "gptq": [
            "optimum[gptq]==1.24.0",
            "gptqmodel<2.1.0",
        ],
        "bitsandbytes": ["bitsandbytes==0.42.0"],
        "awq": ["autoawq>=0.1.0"],
        "all": [
            # Dev dependencies
            "pytest==7.4.0",
            "black==25.1.0",
            "isort==6.0.0",
            "flake8>=6.0.0",
            # Visualization
            "matplotlib==3.10.0",
            "colorama>=0.4.6",
            "jinja2==3.1.5",
            # Data handling
            "datasets==3.4.0",
            "accelerate==1.5.2",
            # Quantization methods
            "optimum[gptq]==1.24.0",
            "gptqmodel<2.1.0",
            "bitsandbytes==0.42.0",
            "autoawq>=0.1.0",
        ],
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
    py_modules=["run_benchmark", "benchmark_model", "visualize_benchmark", "chat_with_model"],
) 