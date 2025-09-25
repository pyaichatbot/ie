#!/usr/bin/env python3
"""
Setup script for Architecture Diagram Parser
Handles virtual environment creation and dependency installation
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    return run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install dependencies from requirements.txt"""
    if platform.system() == "Windows":
        pip_command = "venv\\Scripts\\pip"
    else:
        pip_command = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_command} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    return run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies")

def install_system_dependencies():
    """Provide instructions for system dependencies"""
    print("\n📋 System Dependencies:")
    print("Some OCR engines require system-level dependencies:")
    print()
    
    if platform.system() == "Darwin":  # macOS
        print("🍎 macOS:")
        print("  brew install tesseract")
        print("  # For additional OCR support:")
        print("  brew install poppler")
    elif platform.system() == "Linux":
        print("🐧 Linux (Ubuntu/Debian):")
        print("  sudo apt-get update")
        print("  sudo apt-get install tesseract-ocr")
        print("  sudo apt-get install libtesseract-dev")
        print("  # For additional OCR support:")
        print("  sudo apt-get install poppler-utils")
    elif platform.system() == "Windows":
        print("🪟 Windows:")
        print("  Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Add Tesseract to your PATH environment variable")
    
    print("\n💡 Note: You can skip system dependencies if you only want to use EasyOCR or PaddleOCR")

def create_sample_config():
    """Create a sample configuration file"""
    config_content = """{
  "preprocessing": {
    "upscale_factor": 2.5,
    "upscale_interpolation": 2,
    "enhance_contrast": true,
    "clahe_clip_limit": 3.0,
    "clahe_tile_grid_size": [8, 8],
    "deskew_enabled": true,
    "deskew_angle_threshold": 0.5
  },
  "tiling": {
    "tile_size": 1024,
    "overlap": 128,
    "min_tile_area": 65536
  },
  "ocr": {
    "primary_engine": "easyocr",
    "fallback_engines": ["tesseract", "rapidocr"],
    "confidence_threshold": 0.6,
    "min_text_height": 8,
    "max_text_height": 200
  },
  "arrow_detection": {
    "hough_threshold": 50,
    "min_line_length": 30,
    "max_line_gap": 10,
    "connection_distance_threshold": 20
  },
  "graph_processing": {
    "node_merge_distance": 50,
    "edge_snap_distance": 30,
    "min_node_area": 100
  },
  "diagnostics": {
    "log_level": "INFO",
    "log_file": "architecture_parser.log",
    "performance_tracking": true,
    "save_intermediate_results": false,
    "intermediate_results_dir": "./debug_output"
  }
}"""
    
    with open("config.json", "w") as f:
        f.write(config_content)
    
    print("📄 Created sample configuration file: config.json")

def main():
    """Main setup function"""
    print("🚀 Setting up Architecture Diagram Parser")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create sample config
    create_sample_config()
    
    # Show system dependencies
    install_system_dependencies()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📖 Next steps:")
    print(f"1. Activate virtual environment:")
    print(f"   {get_activation_command()}")
    print("2. Test the installation:")
    print("   python main.py --help")
    print("3. Parse a diagram:")
    print("   python main.py your_diagram.png -o output.json")
    print("\n💡 For debugging, use:")
    print("   python main.py your_diagram.png --debug --save-intermediate")

if __name__ == "__main__":
    main()
