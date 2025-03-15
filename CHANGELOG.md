# Changelog

All notable changes to the Model Quantizer project will be documented in this file.

## [0.3.1] - 2025-03-17

### Fixed
- Fixed GPTQ quantization issue by explicitly installing optimum with GPTQ support (`optimum[gptq]`)
- Updated installation scripts to ensure proper installation of GPTQ dependencies
- Improved requirements-all.txt with explicit GPTQ integration for optimum

### Added
- Confirmed compatibility with Python 3.11 and 3.12 on macOS Sonoma 14.2
- Added detailed platform-specific documentation for macOS, Windows, and Linux
- Enhanced troubleshooting guide with platform-specific information

## [0.3.0] - 2025-03-17

### Fixed
- Fixed dependency installation order issue with gptqmodel requiring torch to be installed first
- Added installation scripts (install_dependencies.sh and install_dependencies.bat) for reliable dependency installation
- Updated requirements-all.txt with clear instructions on installation order
- Improved README.md with detailed installation options and troubleshooting guidance

## [0.2.9] - 2025-03-16

### Added
- Created `requirements-all.txt` for one-shot installation of all dependencies
- Includes all core, quantization, visualization, and data handling packages
- Provides a simple way for users to install everything needed with a single command

## [0.2.8] - 2025-03-16

### Changed
- Further refined dependencies to absolute minimum required set
- Moved optimum from core dependencies to gptq extras
- Improved organization of dependencies in requirements.txt
- Added detailed comments explaining the purpose of each dependency
- Grouped GPTQ dependencies together in the extras

## [0.2.7] - 2025-03-16

### Changed
- Significantly reduced core dependencies to minimize installation issues
- Moved non-essential dependencies to optional extras
- Created new extras: 'viz' for visualization and 'data' for dataset handling
- Changed gptqmodel dependency to use versions below 2.1.0 to avoid numpy>=2.2.2 requirement
- Improved requirements.txt with clearer organization and comments
- Removed torch as a direct dependency to allow more flexible installation

## [0.2.6] - 2025-03-15

### Fixed
- Pinned dependency versions to match working Python 3.11 environment
- Updated torch to version 2.5.1
- Updated bitsandbytes to version 0.42.0
- Updated gptqmodel to version 2.1.0
- Added torchvision and torchaudio as explicit dependencies
- Added "all" extra in setup.py to install all dependencies at once

## [0.2.5] - 2025-03-15

### Fixed
- Added explicit gptqmodel dependency for GPTQ quantization
- Fixed issue with transformers reporting gptqmodel as available when it's not installed
- Added proper requirements.txt file with all dependencies
- Improved dependency management in setup.py

## [0.2.4] - 2025-03-15

### Fixed
- Completely redesigned Python 3.12 compatibility for GPTQ quantization
- Added multiple patching strategies for transformers 4.49.0 compatibility
- Implemented recursive function scanning to find and patch CUDA checks
- Added method-level exception handling to bypass GPU requirements on CPU

## [0.2.3] - 2025-03-15

### Fixed
- Improved Python 3.12 compatibility for GPTQ quantization
- Fixed patch targeting for CUDA availability check in transformers
- Added multiple fallback methods to ensure CPU compatibility

## [0.2.2] - 2025-03-15

### Fixed
- Added Python 3.12 compatibility for GPTQ quantization
- Fixed issue with Optimum's CUDA check in Python 3.12
- Applied monkey patch to bypass CUDA requirement in newer Python versions

## [0.2.1] - 2025-03-15

### Fixed
- Restored GPTQ quantization support for CPU devices
- Removed incorrect device restriction that was blocking CPU usage for GPTQ
- Improved error handling with more helpful suggestions

## [0.2.0] - 2025-03-15

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