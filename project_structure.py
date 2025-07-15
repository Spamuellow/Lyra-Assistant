#!/usr/bin/env python3
"""
AI Assistant Project Structure Setup
Creates the necessary directories and files for the project
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """Create the project directory structure"""
    
    # Base project directory
    project_root = Path.home() / "ai-assistant"
    
    # Create directories
    directories = [
        project_root,
        project_root / "modules",
        project_root / "reference_audio",
        project_root / "logs",
        project_root / "models"  # For storing downloaded models
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        project_root / "modules" / "__init__.py",
    ]
    
    for init_file in init_files:
        init_file.touch()
        print(f"Created file: {init_file}")
    
    print(f"\nProject structure created at: {project_root}")
    print("Next: Run individual module creation scripts")

if __name__ == "__main__":
    create_project_structure()
