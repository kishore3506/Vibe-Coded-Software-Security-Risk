"""
Unit tests for feature extraction and data preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_extraction import (
    extract_code_metrics,
    detect_security_patterns,
    build_feature_dataframe
)
from src.predict import load_inference_artifacts, extract_features, preprocess_input

def test_extract_code_metrics_normal():
    code = """
    int sum_array(int *arr, int len) {
        int total = 0;
        for (int i = 0; i < len; i++) {
            total += arr[i];
        }
        return total;
    }
    """
    metrics = extract_code_metrics(code)
    assert isinstance(metrics, dict)
    assert metrics["line_count"] > 0
    assert metrics["char_count"] > 0
    assert metrics["for_count"] >= 1
    assert metrics["return_count"] >= 1
    assert metrics["array_subscript_count"] >= 1

def test_extract_code_metrics_empty():
    metrics = extract_code_metrics("")
    assert isinstance(metrics, dict)
    assert metrics["char_count"] == 0.0
    assert metrics["line_count"] == 0.0

def test_feature_extraction_pipeline():
    code = "void test() { char *p = malloc(10); free(p); }"
    df = extract_features(code)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert df.shape[1] == 85  # 35 metrics + 50 TF-IDF features

def test_preprocessing_scaling():
    code = "void test() { return; }"
    df = extract_features(code)
    scaled = preprocess_input(df)
    assert isinstance(scaled, np.ndarray)
    assert scaled.shape == (1, 85)
    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()

def test_security_pattern_detection():
    vuln_code = "char b[10]; strcpy(b, src); system(b);"
    patterns = detect_security_patterns(vuln_code)
    assert len(patterns) >= 2
    categories = [p["category"] for p in patterns]
    assert any("Buffer Overflow" in c for c in categories)
    assert any("Command Injection" in c for c in categories)
