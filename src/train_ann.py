"""
Artificial Neural Network (ANN) Training Script.
Builds, trains, and evaluates a deep multi-layer perceptron for security risk detection.
Uses TensorFlow & Keras with Batch Normalization, Dropout, EarlyStopping, and ReduceLROnPlateau.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_preprocessing import load_preprocessed_data

# Set random seeds for reproducibility
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

tf.random.set_seed(42)
np.random.seed(42)

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

def build_ann_model(input_dim: int) -> keras.Model:
    """
    Builds the ANN architecture conforming to academic mini-project specifications:
    Input (input_dim) -> Dense(128, relu) -> BatchNorm -> Dropout(0.3)
                      -> Dense(64, relu)  -> BatchNorm -> Dropout(0.2)
                      -> Dense(32, relu)  -> Dropout(0.2)
                      -> Dense(1, sigmoid)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="Input_Layer"),
        
        layers.Dense(128, activation="relu", name="Dense_128"),
        layers.BatchNormalization(name="BatchNorm_1"),
        layers.Dropout(0.3, name="Dropout_1"),
        
        layers.Dense(64, activation="relu", name="Dense_64"),
        layers.BatchNormalization(name="BatchNorm_2"),
        layers.Dropout(0.2, name="Dropout_2"),
        
        layers.Dense(32, activation="relu", name="Dense_32"),
        layers.Dropout(0.2, name="Dropout_3"),
        
        layers.Dense(1, activation="sigmoid", name="Output_Probability")
    ], name="VibeCode_SecurityRisk_ANN")

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model

def plot_training_history(history, save_path: str):
    """Plot training and validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history.history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Training history plot saved to {save_path}")

def train_and_evaluate_ann(epochs: int = 60, batch_size: int = 32):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("="*60)
    print("TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
    print("="*60)

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_preprocessed_data()
    input_dim = X_train.shape[1]
    print(f"Input Dimension: {input_dim} features")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    # 2. Build model
    model = build_ann_model(input_dim)
    model.summary()

    # 3. Callbacks
    ann_model_path = os.path.join(MODELS_DIR, "ann_model.keras")
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=ann_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    )

    # 4. Train model
    print("\n[INFO] Starting ANN training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_early_stop, cb_checkpoint, cb_reduce_lr],
        verbose=1
    )

    # Save training history plot
    plot_path = os.path.join(FIGURES_DIR, "ann_training_history.png")
    plot_training_history(history, plot_path)

    # 5. Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING BEST ANN ON TEST SET")
    print("="*60)
    best_model = keras.models.load_model(ann_model_path)
    y_prob = best_model.predict(X_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    ann_results = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Confusion_Matrix": cm
    }

    print(f"  ANN Test Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {auc:.4f}")
    print(f"    Confusion Matrix: {cm}")

    # Save metrics JSON
    ann_json_path = os.path.join(METRICS_DIR, "ann_metrics.json")
    with open(ann_json_path, "w", encoding="utf-8") as f:
        json.dump(ann_results, f, indent=4)

    # Save probabilities
    np.save(os.path.join(METRICS_DIR, "ann_test_probs.npy"), y_prob)

    # Update model_comparison.csv
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(csv_path):
        df_comp = pd.read_csv(csv_path, index_col=0)
    else:
        df_comp = pd.DataFrame()

    df_comp.loc["ANN", ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]] = [acc, prec, rec, f1, auc]
    df_comp.to_csv(csv_path)
    print(f"\n[SUCCESS] Updated model comparison table at {csv_path}")

    return ann_results

if __name__ == "__main__":
    train_and_evaluate_ann()
