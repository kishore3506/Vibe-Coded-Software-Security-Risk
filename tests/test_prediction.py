"""
Unit tests for model loading and prediction output format.
"""

import pytest
from src.predict import load_inference_artifacts, predict_security_risk

def test_model_loading():
    model, scaler, tfidf, feature_names = load_inference_artifacts()
    assert model is not None
    assert scaler is not None
    assert tfidf is not None
    assert isinstance(feature_names, list)
    assert len(feature_names) == 85

def test_prediction_output_structure():
    code = """
    void process(char *user) {
        char buf[64];
        strcpy(buf, user);
    }
    """
    res = predict_security_risk(code)
    assert res["status"] == "success"
    assert 0.0 <= res["probability"] <= 1.0
    assert 0.0 <= res["probability_percent"] <= 100.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "POTENTIALLY VULNERABLE" in res["prediction"] or "SECURE" in res["prediction"] or "MODERATE" in res["prediction"]
    assert isinstance(res["pattern_indicators"], list)
    assert "metrics_summary" in res
    assert "disclaimer" in res
    assert "replacement for professional security auditing" in res["disclaimer"]

def test_empty_input_handling():
    res = predict_security_risk("")
    assert "error" in res
    assert res["prediction"] == "EMPTY_INPUT"

def test_whitespace_input_handling():
    res = predict_security_risk("   \n\t  ")
    assert "error" in res
    assert res["prediction"] == "EMPTY_INPUT"

def test_invalid_none_input():
    res = predict_security_risk(None)
    assert "error" in res
    assert res["prediction"] == "INVALID_INPUT"
