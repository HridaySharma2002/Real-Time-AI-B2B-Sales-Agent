#!/usr/bin/env python3
"""
test.py - Fast runner redirecting to tests/test_all.py
"""
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
test_file = os.path.join(root_dir, "tests", "test_all.py")

if __name__ == "__main__":
    if os.path.exists(test_file):
        import runpy
        sys.argv[0] = test_file
        runpy.run_path(test_file, run_name="__main__")
    else:
        print(f"Error: Unified test suite not found at {test_file}")
        sys.exit(1)
