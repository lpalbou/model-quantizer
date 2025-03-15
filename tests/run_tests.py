#!/usr/bin/env python3
"""
Test runner for the quantizer module.
"""

import os
import sys
import unittest

# Add the parent directory to the path to import the quantizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern="test_*.py")
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 