#!/usr/bin/env python3
"""
Environment setup script for AI-Powered Business Intelligence Platform.
This script helps configure the development environment.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup the development environment."""
    
    # Get project root
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    print("🚀 Setting up AI-Powered Business Intelligence Platform Environment")
    print("=" * 70)
    
    # Check if .env exists
    if env_file.exists():
        print("✅ .env file already exists")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response != 'y':
            print("Skipping .env creation")
        else:
            create_env_file(env_file, env_example, project_root)
    else:
        create_env_file(env_file, env_example, project_root)
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    try:
        os.system("pip install -r requirements.txt")
        print("✅ Dependencies installed successfully")
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
    
    # Create necessary directories
    print("\n📁 Creating necessary directories...")
    directories = [
        project_root / "data",
        project_root / "data" / "PDF Folder", 
        project_root / "output",
        project_root / "docs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n" + "=" * 70)
    print("🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Activate your virtual environment: venv\\Scripts\\activate")
    print("2. Set your GROQ_API_KEY in the .env file")
    print("3. Run the application: python -m app.main")
    print("   Or use the run.py script: python run.py")
    print("4. For Docker: docker-compose up -d")

def create_env_file(env_file, env_example, project_root):
    """Create .env file from example."""
    print(f"\n📝 Creating .env file...")
    
    # Get current project path
    project_path = str(project_root)
    
    # Default .env content
    env_content = f"""# Environment Configuration for AI-Powered Business Intelligence Platform

# Add the project root to Python path for proper module imports
PYTHONPATH={project_path}

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/bi_db

# Web Interface Configuration  
WEB_HOST=0.0.0.0
WEB_PORT=7860

# Logging Configuration
LOG_LEVEL=INFO

# AI/ML Configuration
GROQ_API_KEY=your_groq_api_key_here
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created: {env_file}")
    print("⚠️  Please update your GROQ_API_KEY in the .env file")

if __name__ == "__main__":
    setup_environment()
