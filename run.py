#!/usr/bin/env python3
"""
Main entry point for the AI-Powered Business Intelligence Platform.
Run this script from the project root directory.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main application
from app.main import main

if __name__ == "__main__":
    main()
