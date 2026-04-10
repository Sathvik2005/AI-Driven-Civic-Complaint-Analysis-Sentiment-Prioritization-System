#!/usr/bin/env python3
"""
Complete project execution script for Citizen Grievance Analysis System
This script handles:
1. Dataset download (if needed)
2. Model training
3. Service startup (FastAPI + Streamlit)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV_PATH = PROJECT_ROOT / "venv"
MODELS_DIR = PROJECT_ROOT / "trained_models"
DATASETS_DIR = PROJECT_ROOT / "datasets"

def check_requirements():
    """Verify required packages are installed"""
    try:
        import streamlit
        import fastapi
        import sklearn
        import pandas
        import numpy
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nInstalling requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        return True

def download_dataset():
    """Download NYC 311 dataset if not present"""
    csv_files = list(DATASETS_DIR.glob("*.csv"))
    
    if csv_files:
        print(f"✓ Dataset already exists: {csv_files[0]}")
        return True
    
    print("Downloading dataset...")
    try:
        result = subprocess.run(
            [sys.executable, "download_dataset.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✓ Dataset downloaded successfully")
            return True
        else:
            print(f"✗ Dataset download failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Dataset download timed out")
        return False

def train_models():
    """Train models using Week 3 notebook"""
    print("Training models...")
    
    # Check if models already exist
    if (MODELS_DIR / "sentiment_model.joblib").exists():
        print("✓ Models already trained")
        return True
    
    print("Running Week 3 notebook...")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=3600",
                "--output", "WEEK3_IMPROVED_ORIGINAL_DATASET_executed.ipynb",
                "WEEK3_IMPROVED_ORIGINAL_DATASET.ipynb"
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            print("✓ Models trained successfully")
            return True
        else:
            print("✗ Model training failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error training models: {e}")
        return False

def start_services():
    """Start FastAPI and Streamlit services"""
    print("\n" + "=" * 80)
    print("STARTING SERVICES")
    print("=" * 80)
    
    print("\n1. Starting FastAPI server on http://localhost:8000")
    print("   API Documentation: http://localhost:8000/docs")
    
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(2)
    
    print("\n2. Starting Streamlit UI on http://localhost:8501")
    
    ui_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(3)
    
    print("\n" + "=" * 80)
    print("SYSTEM READY")
    print("=" * 80)
    print("\nServices Running:")
    print("  FastAPI:  http://localhost:8000")
    print("  Streamlit: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services...")
    
    try:
        api_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        api_process.terminate()
        ui_process.terminate()
        api_process.wait(timeout=5)
        ui_process.wait(timeout=5)
        print("Services stopped.")

def main():
    """Main execution flow"""
    print("=" * 80)
    print("CITIZEN GRIEVANCE ANALYSIS SYSTEM - STARTUP")
    print("=" * 80)
    
    os.chdir(PROJECT_ROOT)
    
    # Check requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Download dataset
    print("\n2. Ensuring dataset is available...")
    if not download_dataset():
        print("Warning: Dataset download failed, will use synthetic data")
    
    # Train models
    print("\n3. Preparing models...")
    if not train_models():
        sys.exit(1)
    
    # Start services
    print("\n4. Starting services...")
    start_services()

if __name__ == "__main__":
    main()
