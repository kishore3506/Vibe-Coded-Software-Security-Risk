"""
Dataset Download and Preparation Script
Downloads Microsoft CodeXGLUE Defect Detection (Devign benchmark) dataset
from Hugging Face and prepares a reproducible, stratified subset for training.
"""

import os
import urllib.request
import json
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
METADATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_metadata.json")

HF_BASE_URL = "https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection/resolve/main/data"

SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet"
}

def download_file(url: str, dest_path: str):
    """Download a file with progress reporting if not already present."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[INFO] File already exists: {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")
        return
    
    print(f"[INFO] Downloading {url} -> {dest_path}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        chunk_size = 1024 * 1024  # 1 MB chunks
        downloaded = 0
        total_size = int(response.headers.get('Content-Length', 0))
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                print(f"  Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB", end="\r")
        print(f"\n[SUCCESS] Download completed: {dest_path}")

def prepare_dataset(sample_size_train: int = 4200, sample_size_val: int = 900, sample_size_test: int = 900, random_seed: int = 42):
    """
    Downloads the CodeXGLUE Devign dataset and creates a reproducible stratified subset.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Download raw parquet files
    raw_paths = {}
    for split, filename in SPLIT_FILES.items():
        dest = os.path.join(RAW_DIR, f"{split}.parquet")
        url = f"{HF_BASE_URL}/{filename}"
        download_file(url, dest)
        raw_paths[split] = dest

    print("\n" + "="*60)
    print("ANALYZING FULL ORIGINAL DATASET")
    print("="*60)
    
    full_stats = {}
    dfs = {}
    for split, path in raw_paths.items():
        df = pd.read_parquet(path)
        # Standardize target as integer (0: Secure, 1: Vulnerable)
        df['target'] = df['target'].astype(int)
        dfs[split] = df
        
        counts = df['target'].value_counts().to_dict()
        full_stats[split] = {
            "total_samples": int(len(df)),
            "secure_count (0)": int(counts.get(0, 0)),
            "vulnerable_count (1)": int(counts.get(1, 0)),
            "vulnerability_rate": float(counts.get(1, 0) / len(df))
        }
        print(f"Split [{split.upper()}]: Total = {len(df):,}, Secure (0) = {counts.get(0, 0):,}, Vulnerable (1) = {counts.get(1, 0):,} ({counts.get(1, 0)/len(df)*100:.2f}%)")

    total_orig = sum(s["total_samples"] for s in full_stats.values())
    print(f"Total Original Devign Dataset Size: {total_orig:,} functions\n")

    print("="*60)
    print("CREATING REPRODUCIBLE STRATIFIED SUBSET FOR FAST, EFFICIENT EXPERIMENTATION")
    print("="*60)

    # Subsampling with stratification to preserve exact class ratio
    subsets = {}
    subset_targets = {
        "train": sample_size_train,
        "validation": sample_size_val,
        "test": sample_size_test
    }

    subset_stats = {}
    for split, df in dfs.items():
        target_size = min(subset_targets[split], len(df))
        # Stratified sample
        subset_df, _ = train_test_split(
            df,
            train_size=target_size,
            stratify=df['target'],
            random_state=random_seed
        )
        subset_df = subset_df.reset_index(drop=True)
        subsets[split] = subset_df
        
        # Save to processed directory
        out_csv = os.path.join(PROCESSED_DIR, f"{split}_subset.csv")
        out_parquet = os.path.join(PROCESSED_DIR, f"{split}_subset.parquet")
        subset_df.to_csv(out_csv, index=False)
        subset_df.to_parquet(out_parquet, index=False)
        
        counts = subset_df['target'].value_counts().to_dict()
        subset_stats[split] = {
            "subset_samples": int(len(subset_df)),
            "secure_count (0)": int(counts.get(0, 0)),
            "vulnerable_count (1)": int(counts.get(1, 0)),
            "vulnerability_rate": float(counts.get(1, 0) / len(subset_df))
        }
        print(f"Subset [{split.upper()}]: Total = {len(subset_df):,}, Secure (0) = {counts.get(0, 0):,}, Vulnerable (1) = {counts.get(1, 0):,} ({counts.get(1, 0)/len(subset_df)*100:.2f}%)")

    total_subset = sum(s["subset_samples"] for s in subset_stats.values())
    print(f"\nTotal Subset Size: {total_subset:,} functions ({sample_size_train} train, {sample_size_val} val, {sample_size_test} test)")
    
    # Save metadata
    metadata = {
        "dataset_name": "Microsoft CodeXGLUE Defect Detection (Devign benchmark)",
        "source_url": "https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection",
        "original_paper": "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks (Zhou et al., NeurIPS 2019)",
        "programming_languages": ["C", "C++"],
        "task_type": "Binary Vulnerability Classification (0=Secure, 1=Vulnerable)",
        "original_statistics": full_stats,
        "total_original_samples": total_orig,
        "subset_statistics": subset_stats,
        "total_subset_samples": total_subset,
        "random_seed": random_seed,
        "sampling_method": "Stratified random sampling preserving exact class distribution"
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUCCESS] Dataset metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    prepare_dataset()
