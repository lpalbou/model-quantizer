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
        ],
    },
) 