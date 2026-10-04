"""
Baseline Model Training Script.
Trains Logistic Regression, Random Forest, Support Vector Machine (and Gradient Boosting).
Evaluates on the test split and saves metrics and prediction probabilities for comparative analysis.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preprocessing import load_preprocessed_data

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

def train_and_evaluate_baselines():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("="*60)
    print("TRAINING BASELINE MACHINE LEARNING MODELS")
    print("="*60)

    # 1. Load preprocessed train, val, test data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_preprocessed_data()
    print(f"Features: {len(feature_names)}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Define baseline classifiers
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            solver='lbfgs'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    }

    results = {}
    test_probs = {}

    for name, model in models.items():
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train, y_train)

        # Predict on test split
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        test_probs[name] = y_prob

        # Calculate metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        }

        print(f"  {name} Results:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}")

        # Save model artifact
        slug = name.lower().replace(" ", "_")
        model_save_path = os.path.join(MODELS_DIR, f"baseline_{slug}.pkl")
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)

    # Save baseline metrics JSON
    baseline_json_path = os.path.join(METRICS_DIR, "baseline_metrics.json")
    with open(baseline_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] Baseline metrics saved to {baseline_json_path}")

    # Save baseline test probabilities for plotting ROC curves
    np.savez(
        os.path.join(METRICS_DIR, "baseline_test_probs.npz"),
        **{name.replace(" ", "_"): prob for name, prob in test_probs.items()}
    )

    # Update or create model_comparison.csv
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = "Model"
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_results.to_csv(csv_path)
    print(f"[SUCCESS] Model comparison updated at {csv_path}")

    return results

if __name__ == "__main__":
    train_and_evaluate_baselines()
