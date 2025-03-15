# Changelog

All notable changes to the Model Quantizer project will be documented in this file.

## [0.2.0] - 2023-07-15

### Added
- PyPI package support with `model-quantizer` now available via pip
- Automatic model card generation when quantizing models
- Ability to update model cards with benchmark results
- New command-line tools with consistent naming:
  - `model-quantizer`: Main tool for quantizing models
  - `benchmark-model`: Tool for benchmarking models
  - `run-benchmark`: All-in-one benchmarking solution
  - `visualize-benchmark`: Tool for creating visual benchmark reports
  - `chat-with-model`: Interactive testing tool

### Changed
- Renamed `benchmark_your_model.py` to `benchmark_model.py`
- Converted `run_benchmark.sh` to `run_benchmark.py` for cross-platform compatibility
- Updated `chat_with_model.py` to accept a single `model_path` parameter
- Updated `setup.py` to include necessary dependencies for all command-line tools
- Consolidated publishing documentation into a single comprehensive guide
- Improved documentation structure with clearer workflow steps
- Enhanced examples with more detailed instructions and use cases

### Fixed
- Cross-platform compatibility issues, especially for macOS users
- Inconsistent command-line interfaces across tools
- Missing dependencies in setup.py

## [0.1.0] - 2023-06-01

### Added
- Initial release of Model Quantizer
- Support for GPTQ, BitsAndBytes, and AWQ quantization methods
- Basic benchmarking capabilities
- Interactive testing via chat interface
- Documentation for quantizing models
- Example scripts for common use cases 