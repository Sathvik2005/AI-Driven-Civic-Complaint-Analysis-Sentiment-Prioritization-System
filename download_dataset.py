"""
NYC 311 Service Requests - Complete Dataset Downloader
Downloads the full 14.69GB dataset from Kaggle
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_config.exists():
        print("❌ Kaggle credentials not found!")
        print("\nTo download the dataset, you need to:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Save the kaggle.json file to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json (on Unix)")
        return False
    return True


def download_from_kaggle(dataset_name, output_dir):
    """Download dataset from Kaggle"""
    print(f"\n📥 Starting download of {dataset_name}...")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        cmd = f'kaggle datasets download -d {dataset_name} -p {output_dir} --unzip'
        print(f"🔄 Running: {cmd}\n")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Download completed successfully!")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during download: {str(e)}")
        return False


def verify_dataset(csv_path):
    """Verify dataset after download"""
    try:
        print("\n🔍 Verifying dataset...")
        
        # Check file exists
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return False
        
        # Check file size
        file_size_gb = os.path.getsize(csv_path) / (1024**3)
        print(f"📊 File size: {file_size_gb:.2f} GB")
        
        # Read first few rows to verify structure
        print("📋 Reading dataset structure...")
        df = pd.read_csv(csv_path, nrows=100)
        
        print(f"\n✅ Dataset verified!")
        print(f"   Columns: {df.shape[1]}")
        print(f"   Sample rows loaded: {df.shape[0]}")
        print(f"\nColumn names:")
        for col in df.columns:
            print(f"   - {col}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification error: {str(e)}")
        return False


def get_dataset_info(csv_path):
    """Get basic info about the dataset"""
    try:
        print("\n📈 Getting dataset information...")
        print("   (Using sampling to avoid loading entire 14GB file)")
        
        # Sample 10,000 rows
        chunk_size = 100000
        total_rows = 0
        sample_data = []
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            total_rows += len(chunk)
            if len(sample_data) < 10000:
                sample_data.append(chunk)
            
            if len(sample_data) > 0:
                break  # Just get first chunk for stats
        
        if sample_data:
            df_sample = pd.concat(sample_data, ignore_index=True)
            print(f"\n📊 Dataset Statistics (from sample):")
            print(f"   Est. Total Rows: {total_rows:,}+")
            print(f"   Columns: {len(df_sample.columns)}")
            print(f"\n   Key Columns:")
            for col in ['Created Date', 'Agency', 'Complaint Type', 'Descriptor']:
                if col in df_sample.columns:
                    unique_count = df_sample[col].nunique()
                    print(f"   - {col}: {unique_count:,} unique values")
        
    except Exception as e:
        print(f"⚠️  Could not get full stats: {str(e)}")


def main():
    """Main download workflow"""
    print("\n" + "="*70)
    print("NYC 311 SERVICE REQUESTS - COMPLETE DATASET DOWNLOADER")
    print("="*70)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📦 Dataset: NYC 311 Service Requests (2010-Present)")
    print(f"💾 Size: ~14.69 GB (21.73 million records)")
    print(f"⏱️  Estimated Download Time: 30-60 minutes (depends on internet speed)")
    
    # Setup
    project_root = Path.cwd()
    datasets_dir = project_root / 'datasets'
    datasets_dir.mkdir(exist_ok=True)
    
    csv_path = datasets_dir / '311-service-requests-from-2010-to-present.csv'
    
    # Check if already downloaded
    if csv_path.exists():
        file_size_gb = os.path.getsize(csv_path) / (1024**3)
        print(f"\n✅ Dataset already exists: {csv_path}")
        print(f"   File size: {file_size_gb:.2f} GB")
        
        response = input("\n❓ Run verification? (y/n): ").strip().lower()
        if response == 'y':
            get_dataset_info(str(csv_path))
        return csv_path
    
    # Check Kaggle API
    print("\n🔐 Checking Kaggle API credentials...")
    if not check_kaggle_credentials():
        print("\n⚠️  Cannot proceed without Kaggle credentials")
        print("Please configure Kaggle API and try again.")
        return None
    
    # Download
    success = download_from_kaggle(
        'cityofnewyork/nyc-311-calls',
        str(datasets_dir)
    )
    
    if not success:
        return None
    
    # Verify
    if verify_dataset(str(csv_path)):
        get_dataset_info(str(csv_path))
        print(f"\n✅ Dataset ready for training!")
        print(f"   Location: {csv_path}")
        return csv_path
    
    return None


if __name__ == '__main__':
    try:
        dataset_path = main()
        if dataset_path:
            print("\n" + "="*70)
            print("✅ DOWNLOAD COMPLETE - Ready for Week 3 training")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("❌ DOWNLOAD FAILED - Please check the errors above")
            print("="*70)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
