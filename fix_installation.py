#!/usr/bin/env python3
"""
Script to fix the run-benchmark installation issue.
"""

import os
import sys
import shutil
import site
import subprocess

def main():
    """Fix the run-benchmark installation issue."""
    print("Fixing run-benchmark installation issue...")
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    print(f"Site-packages directory: {site_packages}")
    
    # Check if run_benchmark.py exists in the current directory
    if not os.path.exists("run_benchmark.py"):
        print("Error: run_benchmark.py not found in the current directory.")
        print("Please run this script from the model-quantizer repository root.")
        return 1
    
    # Copy run_benchmark.py to the site-packages directory
    print("Copying run_benchmark.py to site-packages...")
    shutil.copy("run_benchmark.py", site_packages)
    print("Done!")
    
    # Reinstall the package
    print("Reinstalling the package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    print("Installation fixed!")
    print("\nYou should now be able to run the run-benchmark command.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 