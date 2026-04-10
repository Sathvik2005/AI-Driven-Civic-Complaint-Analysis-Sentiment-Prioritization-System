#!/usr/bin/env python3
"""
Download NYC 311 Service Requests dataset from Kaggle
"""

import os
import sys
from pathlib import Path
import subprocess
import json

PROJECT_ROOT = Path(__file__).parent
KAGGLE_CONFIG = PROJECT_ROOT / "kaggle.json"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

def setup_kaggle():
    """Setup Kaggle credentials"""
    if not KAGGLE_CONFIG.exists():
        print("Error: kaggle.json not found in project root")
        print("Download it from: https://www.kaggle.com/settings/account")
        return False
    
    # Copy to Kaggle config directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(KAGGLE_CONFIG, kaggle_dir / "kaggle.json")
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
    print(f"Kaggle credentials configured at {kaggle_dir / 'kaggle.json'}")
    return True

def download_dataset():
    """Download NYC 311 dataset"""
    dataset_id = "sobhanmoosavi/us-accidents"  # Alternative: NYC 311 specific dataset
    
    # Try main NYC 311 dataset first
    main_dataset = "new-york-city-311-service-requests-2023"
    
    try:
        print(f"Downloading {main_dataset}...")
        cmd = [
            "kaggle", "datasets", "download",
            "-d", main_dataset,
            "-p", str(DATASETS_DIR),
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print(f"Successfully downloaded to {DATASETS_DIR}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nTrying alternative approach...")
        return False

def create_fallback_dataset():
    """Create a synthetic NYC 311-like dataset if download fails"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    agencies = [
        "NYPD", "DOE", "DEP", "DOT", "DSNY", "FDNY", "HPD", "DOHMH",
        "Con Edison", "NYCHA", "311", "DCP", "SCA", "DPR"
    ]
    
    complaint_types = [
        "Blocked Driveway", "Noise - Residential", "Illegal Parking",
        "Pothole", "Unsanitary Condition", "Street Sign - Dangling",
        "Water System", "Graffiti", "Rodent Activity", "Street Condition",
        "Heating", "Plumbing", "Paint - Plaster", "Trash Collection",
        "Street Light Condition", "Traffic Signal Condition",
        "Water Leak - Main", "Flooding", "Construction", "Sidewalk Condition"
    ]
    
    complaint_descriptors = [
        "Urgent attention needed",
        "Multiple reports received",
        "High priority",
        "Routine maintenance",
        "Civilian complaint",
        "Safety hazard",
        "Infrastructure damage",
        "Environmental concern",
        "Public health issue",
        "Regular complaint"
    ]
    
    n_records = 50000
    
    data = {
        "Unique ID": range(1, n_records + 1),
        "Created Date": pd.date_range(start="2020-01-01", periods=n_records, freq="15min"),
        "Complaint Type": np.random.choice(complaint_types, n_records),
        "Descriptor": np.random.choice(complaint_descriptors, n_records),
        "Agency Name": np.random.choice(agencies, n_records),
        "Agency": np.random.choice(agencies, n_records),
        "Borough": np.random.choice(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n_records),
        "Status": np.random.choice(["Open", "Closed"], n_records, p=[0.3, 0.7]),
    }
    
    df = pd.DataFrame(data)
    
    # Save dataset
    output_path = DATASETS_DIR / "311-service-requests.csv"
    df.to_csv(output_path, index=False)
    print(f"Created fallback dataset: {output_path}")
    print(f"Records: {len(df):,}")
    return str(output_path)

if __name__ == "__main__":
    print("=" * 80)
    print("NYC 311 DATASET DOWNLOADER")
    print("=" * 80)
    
    # Setup Kaggle
    if setup_kaggle():
        if not download_dataset():
            print("\nNo internet connection or Kaggle API issue.")
            print("Creating synthetic dataset for development...\n")
            create_fallback_dataset()
    else:
        print("Creating synthetic dataset for development...\n")
        create_fallback_dataset()
    
    # Verify dataset exists
    csvs = list(DATASETS_DIR.glob("*.csv"))
    if csvs:
        print(f"\nDataset ready: {csvs[0]}")
        print(f"Size: {csvs[0].stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("Error: No dataset found")
        sys.exit(1)
