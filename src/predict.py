"""
Inference & Risk Prediction Module for Vibe-Code Security Analysis.
Exposes extract_features, preprocess_input, and predict_security_risk.
Strict security safeguards: input validation, max size limit, NO code execution.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_extraction import extract_code_metrics, detect_security_patterns, build_feature_dataframe

MODELS_DIR = os.path.join(BASE_DIR, "models")
ANN_MODEL_PATH = os.path.join(MODELS_DIR, "ann_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# Security constants
MAX_CODE_LENGTH = 50000  # 50 KB text limit to protect against DoS

_MODEL_CACHE = {
    "ann": None,
    "scaler": None,
    "tfidf": None,
    "feature_names": None
}

def load_inference_artifacts():
    """Lazy loader for model and preprocessing artifacts."""
    if _MODEL_CACHE["scaler"] is None:
        if not os.path.exists(SCALER_PATH) or not os.path.exists(TFIDF_PATH):
            raise FileNotFoundError(f"Preprocessing artifacts missing in {MODELS_DIR}. Run preprocessing and training first.")
        with open(SCALER_PATH, "rb") as f:
            _MODEL_CACHE["scaler"] = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            _MODEL_CACHE["tfidf"] = pickle.load(f)
        with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            _MODEL_CACHE["feature_names"] = json.load(f)

    if _MODEL_CACHE["ann"] is None:
        if not os.path.exists(ANN_MODEL_PATH):
            raise FileNotFoundError(f"Trained ANN model not found at {ANN_MODEL_PATH}. Train the model first.")
        from tensorflow import keras
        _MODEL_CACHE["ann"] = keras.models.load_model(ANN_MODEL_PATH)

    return _MODEL_CACHE["ann"], _MODEL_CACHE["scaler"], _MODEL_CACHE["tfidf"], _MODEL_CACHE["feature_names"]

def extract_features(code: str) -> pd.DataFrame:
    """
    Extracts numerical features and TF-IDF features using the trained TF-IDF vectorizer.
    """
    _, _, tfidf, _ = load_inference_artifacts()
    df, _ = build_feature_dataframe([code], tfidf)
    return df

def preprocess_input(features_df: pd.DataFrame) -> np.ndarray:
    """
    Scales features using the fitted StandardScaler.
    """
    _, scaler, _, _ = load_inference_artifacts()
    return scaler.transform(features_df)

def predict_security_risk(code: str) -> dict:
    """
    Analyzes a raw code snippet, extracts security metrics and TF-IDF features,
    and predicts vulnerability risk probability using the trained Artificial Neural Network.

    Security Controls:
    - Input length limit (< 50,000 characters)
    - Rejection of empty/null inputs
    - ZERO code execution (no eval, exec, subprocess)
    """
    # 1. Secure Input Validation
    if code is None or not isinstance(code, str):
        return {
            "error": "Input must be a valid non-empty string.",
            "probability": 0.0,
            "risk_level": "UNKNOWN",
            "prediction": "INVALID_INPUT",
            "pattern_indicators": [],
            "disclaimer": "Invalid input provided."
        }

    if len(code) > MAX_CODE_LENGTH:
        return {
            "error": f"Input size ({len(code):,} chars) exceeds maximum allowable length ({MAX_CODE_LENGTH:,} chars).",
            "probability": 0.0,
            "risk_level": "UNKNOWN",
            "prediction": "OVERSIZED_INPUT",
            "pattern_indicators": [],
            "disclaimer": "Request rejected due to size limits."
        }

    stripped_code = code.strip()
    if not stripped_code:
        return {
            "error": "Empty code snippet provided. Please enter source code to analyze.",
            "probability": 0.0,
            "risk_level": "UNKNOWN",
            "prediction": "EMPTY_INPUT",
            "pattern_indicators": [],
            "disclaimer": "No code provided."
        }

    # 2. Extract features & preprocess
    ann_model, _, _, _ = load_inference_artifacts()
    features_df = extract_features(code)
    scaled_vector = preprocess_input(features_df)

    # 3. ANN Model Inference
    probability = float(ann_model.predict(scaled_vector, verbose=0)[0][0])

    # 4. Determine Risk Level
    if probability >= 0.60:
        risk_level = "HIGH"
        predicted_label = "POTENTIALLY VULNERABLE (SECURITY RISK)"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
        predicted_label = "MODERATE SECURITY RISK"
    else:
        risk_level = "LOW"
        predicted_label = "LIKELY SECURE (LOW RISK)"

    # 5. Rule-based security pattern indicators (distinct heuristic screening)
    pattern_indicators = detect_security_patterns(code)

    # 6. Structured Result
    explanation = (
        "The Artificial Neural Network detected statistical and syntactic patterns "
        "strongly correlated with software vulnerabilities in the benchmark repository."
        if probability >= 0.50 else
        "The Artificial Neural Network did not identify dominant vulnerability signatures "
        "in the extracted lexical and structural feature representation."
    )

    disclaimer = (
        "This prediction is generated by a machine-learning model (Artificial Neural Network) "
        "and should not be treated as a replacement for professional security auditing, "
        "symbolic execution, or dynamic fuzzing."
    )

    return {
        "status": "success",
        "probability": probability,
        "probability_percent": round(probability * 100, 2),
        "risk_level": risk_level,
        "prediction": predicted_label,
        "explanation": explanation,
        "pattern_indicators": pattern_indicators,
        "feature_count": scaled_vector.shape[1],
        "metrics_summary": {
            "lines": int(code.count('\n') + 1),
            "characters": len(code),
            "branching_nodes": int(features_df.get("total_branching", [0])[0]),
            "buffer_operations": int(features_df.get("buffer_unsafe_count", [0])[0]),
            "command_invocations": int(features_df.get("command_exec_count", [0])[0])
        },
        "disclaimer": disclaimer
    }

if __name__ == "__main__":
    sample_vuln = '''
    void process_packet(char *user_input) {
        char buffer[128];
        strcpy(buffer, user_input);
        system(buffer);
    }
    '''
    res = predict_security_risk(sample_vuln)
    print("\n--- SAMPLE INFERENCE TEST ---")
    print(json.dumps(res, indent=2))
