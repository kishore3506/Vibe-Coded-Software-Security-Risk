"""
Data Preprocessing Pipeline.
Fits Scaler and TF-IDF strictly on the training partition to prevent data leakage.
Saves preprocessing artifacts to models/ directory.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_extraction import (
    fit_tfidf_vectorizer,
    build_feature_dataframe,
    extract_code_metrics
)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

def load_split_data(split_name: str) -> pd.DataFrame:
    """Load a dataset split from parquet or CSV."""
    parquet_path = os.path.join(PROCESSED_DIR, f"{split_name}_subset.parquet")
    csv_path = os.path.join(PROCESSED_DIR, f"{split_name}_subset.csv")
    
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Split file for {split_name} not found in {PROCESSED_DIR}")
    
    # Handle missing values
    df['func'] = df['func'].fillna("")
    df['target'] = df['target'].astype(int)
    return df

def run_preprocessing_pipeline(tfidf_max_features: int = 50):
    """
    Executes the full preprocessing pipeline:
    1. Loads train, val, test subsets
    2. Fits TF-IDF on train code
    3. Builds feature matrices
    4. Fits StandardScaler on train features
    5. Transforms val and test sets
    6. Saves artifacts and preprocessed numpy matrices
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("="*60)
    print("RUNNING PREPROCESSING PIPELINE (LEAKAGE-FREE)")
    print("="*60)

    # 1. Load splits
    train_df = load_split_data("train")
    val_df = load_split_data("validation")
    test_df = load_split_data("test")

    print(f"Loaded Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples.")

    # 2. Fit TF-IDF on training partition only
    print(f"[INFO] Fitting TF-IDF vectorizer (max_features={tfidf_max_features}) strictly on train...")
    tfidf_vectorizer = fit_tfidf_vectorizer(train_df['func'].tolist(), max_features=tfidf_max_features)

    # 3. Build features
    print("[INFO] Building feature matrices for train, val, and test...")
    X_train_df, feature_names = build_feature_dataframe(train_df['func'].tolist(), tfidf_vectorizer)
    X_val_df, _ = build_feature_dataframe(val_df['func'].tolist(), tfidf_vectorizer)
    X_test_df, _ = build_feature_dataframe(test_df['func'].tolist(), tfidf_vectorizer)

    y_train = train_df['target'].to_numpy(dtype=np.int32)
    y_val = val_df['target'].to_numpy(dtype=np.int32)
    y_test = test_df['target'].to_numpy(dtype=np.int32)

    print(f"Extracted {len(feature_names)} features per sample.")

    # 4. Fit StandardScaler on training features only
    print("[INFO] Fitting StandardScaler strictly on train features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    # 5. Save preprocessing artifacts
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=4)

    print(f"[SUCCESS] Saved artifacts: scaler, tfidf, feature_names to {MODELS_DIR}")

    # 6. Save matrices to data/processed for instant model training
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val_scaled)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"[SUCCESS] Preprocessed matrices saved to {PROCESSED_DIR}")
    print(f"X_train shape: {X_train_scaled.shape}, y_train class balance: {np.bincount(y_train)}")
    print(f"X_val shape:   {X_val_scaled.shape}, y_val class balance:   {np.bincount(y_val)}")
    print(f"X_test shape:  {X_test_scaled.shape}, y_test class balance:  {np.bincount(y_test)}")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_names

def load_preprocessed_data():
    """Helper to load preprocessed npy arrays."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

def preprocess_single_code(code: str) -> np.ndarray:
    """
    Transforms a single source code snippet using saved preprocessing artifacts
    into the exact scaled feature vector required by trained models.
    """
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    
    df, _ = build_feature_dataframe([code], tfidf)
    scaled_vector = scaler.transform(df)
    return scaled_vector

if __name__ == "__main__":
    run_preprocessing_pipeline()
