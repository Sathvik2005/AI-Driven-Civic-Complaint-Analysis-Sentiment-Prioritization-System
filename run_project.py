"""
NYC 311 Citizen Grievance Analysis - Complete Project Orchestrator
Handles: Dataset Download → Model Training → API/UI Deployment
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import argparse

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASETS_DIR = PROJECT_ROOT / 'datasets'
MODELS_DIR = PROJECT_ROOT / 'trained_models'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create necessary directories
for dir_path in [DATASETS_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)


def log(message, level="INFO"):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    
    # Also write to log file
    with open(LOGS_DIR / "project.log", "a") as f:
        f.write(log_message + "\n")


def banner(text):
    """Print formatted banner"""
    width = 80
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def check_dependencies():
    """Verify all required packages are installed"""
    log("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'joblib',
        'matplotlib', 'seaborn', 'nltk'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            log(f"  ✓ {package}", "OK")
        except ImportError:
            log(f"  ✗ {package} NOT FOUND", "ERROR")
            missing.append(package)
    
    if missing:
        log(f"\nMissing packages: {', '.join(missing)}", "ERROR")
        log("Install with: pip install -r requirements.txt", "INFO")
        return False
    
    return True


def download_dataset():
    """Download the complete NYC 311 dataset"""
    banner("STEP 1: DATASET DOWNLOAD")
    
    csv_file = DATASETS_DIR / '311-service-requests-from-2010-to-present.csv'
    
    if csv_file.exists():
        size_gb = csv_file.stat().st_size / (1024**3)
        log(f"Dataset already exists: {csv_file} ({size_gb:.2f} GB)")
        return True
    
    log("Starting dataset download...")
    log(f"Destination: {DATASETS_DIR}")
    log("Note: This is a 14.69GB file. Download may take 30-60 minutes.")
    
    try:
        result = subprocess.run([
            sys.executable, "download_dataset.py"
        ], cwd=PROJECT_ROOT, capture_output=False, timeout=7200)
        
        if result.returncode != 0:
            log("Dataset download failed", "ERROR")
            return False
        
        if not csv_file.exists():
            log("Dataset file not found after download", "ERROR")
            return False
        
        size_gb = csv_file.stat().st_size / (1024**3)
        log(f"✓ Dataset downloaded successfully ({size_gb:.2f} GB)", "OK")
        return True
        
    except Exception as e:
        log(f"Error downloading dataset: {str(e)}", "ERROR")
        return False


def train_models_on_complete_dataset():
    """Train models on the complete dataset"""
    banner("STEP 2: MODEL TRAINING ON COMPLETE DATASET")
    
    csv_file = DATASETS_DIR / '311-service-requests-from-2010-to-present.csv'
    
    if not csv_file.exists():
        log(f"Dataset not found: {csv_file}", "ERROR")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        import joblib
        import warnings
        warnings.filterwarnings('ignore')
        
        log("Loading complete dataset...")
        df = pd.read_csv(csv_file)
        total_rows = len(df)
        log(f"✓ Loaded {total_rows:,} records", "OK")
        
        # Data preprocessing
        log("Preprocessing data...")
        df['Complaint_Text'] = df['Complaint Type'].fillna('') + ' ' + df['Descriptor'].fillna('')
        df['Complaint_Text'] = df['Complaint_Text'].str.lower().str.strip()
        
        # Remove empty texts
        df = df[df['Complaint_Text'].str.len() > 0]
        log(f"✓ Records after cleaning: {len(df):,}")
        
        # Derive sentiment labels (same logic as Week 3)
        def derive_sentiment(text):
            critical_keywords = ['urgent', 'immediate', 'critical', 'danger', 'safety', 'emergency', 'severe', 'flooding', 'leak']
            negative_keywords = ['poor', 'bad', 'broken', 'damaged', 'issue', 'problem']
            positive_keywords = ['good', 'great', 'excellent', 'fixed']
            
            text = str(text).lower()
            if any(k in text for k in critical_keywords):
                return 'Critical'
            elif any(k in text for k in negative_keywords):
                return 'Negative'
            elif any(k in text for k in positive_keywords):
                return 'Positive'
            return 'Neutral'
        
        log("Deriving sentiment labels...")
        df['sentiment'] = df['Complaint_Text'].apply(derive_sentiment)
        
        # Use stratified sample for training (to manage memory)
        sample_size = min(1000000, len(df))  # Max 1M records for training
        log(f"Sampling {sample_size:,} records for training...")
        df_train = df.sample(n=sample_size, random_state=42, stratify=df['sentiment'])
        
        X = df_train['Complaint_Text'].values
        y = df_train['sentiment'].values
        
        # Split data
        log("Splitting data into train/test sets (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        log(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # TF-IDF Vectorization
        log("Training TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        log(f"✓ TF-IDF vocab size: {len(vectorizer.get_feature_names_out()):,}")
        
        # Train LinearSVC classifier
        log("Training LinearSVC classifier...")
        classifier = LinearSVC(random_state=42, max_iter=2000, verbose=0)
        classifier.fit(X_train_tfidf, y_train)
        log("✓ Model training complete")
        
        # Evaluate
        log("Evaluating model...")
        y_train_pred = classifier.predict(X_train_tfidf)
        y_test_pred = classifier.predict(X_test_tfidf)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='macro')
        
        log(f"  Train Accuracy: {train_acc:.4f}")
        log(f"  Test Accuracy: {test_acc:.4f}")
        log(f"  F1-Score (macro): {f1:.4f}")
        
        # Save models
        log("Saving models...")
        joblib.dump(classifier, MODELS_DIR / 'sentiment_model_full_dataset.joblib')
        joblib.dump(vectorizer, MODELS_DIR / 'tfidf_vectorizer_full_dataset.joblib')
        
        # Save metadata
        metadata = {
            'total_records_in_dataset': total_rows,
            'records_used_for_training': len(df_train),
            'train_records': len(X_train),
            'test_records': len(X_test),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'f1_macro': float(f1),
            'classes': list(classifier.classes_),
            'vocab_size': len(vectorizer.get_feature_names_out()),
            'training_date': datetime.now().isoformat()
        }
        
        import json
        with open(MODELS_DIR / 'model_metadata_full_dataset.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log("✓ Models saved successfully", "OK")
        log(f"  classifier: {MODELS_DIR / 'sentiment_model_full_dataset.joblib'}")
        log(f"  vectorizer: {MODELS_DIR / 'tfidf_vectorizer_full_dataset.joblib'}")
        
        return True
        
    except Exception as e:
        log(f"Error during model training: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def start_services(api_port=8000, streamlit_port=8501):
    """Start API and UI services"""
    banner("STEP 3: STARTING SERVICES")
    
    log(f"API Server: http://localhost:{api_port}")
    log(f"Streamlit UI: http://localhost:{streamlit_port}")
    
    try:
        # Check if Docker is being used
        docker_compose_path = PROJECT_ROOT / 'docker-compose.yml'
        if docker_compose_path.exists():
            log("Starting with Docker Compose...")
            result = subprocess.run([
                'docker-compose', 'up'
            ], cwd=PROJECT_ROOT)
            return result.returncode == 0
        else:
            log("Starting with Python directly...")
            log("Note: Please ensure FastAPI server and Streamlit are running separately")
            return True
            
    except Exception as e:
        log(f"Error starting services: {str(e)}", "ERROR")
        return False


def main():
    """Main orchestration workflow"""
    banner("NYC 311 CITIZEN GRIEVANCE ANALYSIS - COMPLETE PROJECT")
    
    parser = argparse.ArgumentParser(description="Run the NYC 311 Analysis Project")
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    parser.add_argument('--api-port', type=int, default=8000, help='API port (default: 8000)')
    parser.add_argument('--ui-port', type=int, default=8501, help='Streamlit port (default: 8501)')
    parser.add_argument('--start-services', action='store_true', help='Start API and UI services')
    
    args = parser.parse_args()
    
    log(f"Project Root: {PROJECT_ROOT}")
    log(f"Python: {sys.version.split()[0]}")
    
    # Check dependencies
    if not check_dependencies():
        log("Please install missing dependencies", "ERROR")
        return 1
    
    # Step 1: Download dataset
    if not args.skip_download:
        if not download_dataset():
            log("Dataset download failed", "ERROR")
            return 1
    else:
        log("Skipping dataset download (--skip-download)")
    
    # Step 2: Train models
    if not args.skip_train:
        if not train_models_on_complete_dataset():
            log("Model training failed", "ERROR")
            return 1
    else:
        log("Skipping model training (--skip-train)")
    
    # Step 3: Start services
    if args.start_services:
        if not start_services(args.api_port, args.ui_port):
            log("Failed to start services", "ERROR")
            return 1
    
    banner("PROJECT COMPLETION STATUS")
    log("✓ All steps completed successfully!", "OK")
    log(f"\nNext steps:")
    log(f"  1. API available at: http://localhost:{args.api_port}")
    log(f"  2. UI available at: http://localhost:{args.ui_port}")
    log(f"  3. Models saved in: {MODELS_DIR}")
    log(f"  4. Check logs: {LOGS_DIR}/project.log")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log("\n\nProject interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
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
