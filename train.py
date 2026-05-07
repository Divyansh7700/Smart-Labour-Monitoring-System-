"""
train.py — Training script for the Labour Activity Monitoring CNN model.

This script:
  1. Loads accelerometer CSV datasets (6 working + 2 not-working activities)
  2. Creates sliding windows of 50 time-steps
  3. Splits data into train/test sets
  4. Normalises features with StandardScaler (saved as scaler.pkl)
  5. Builds and trains a 1D CNN (Conv1D) for binary classification
  6. Saves the trained model as model.keras
  7. Prints test accuracy

Usage:
    python train.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ============================
# Configuration
# ============================
BASE_DIR = r"C:\Users\Divyansh\Downloads\project"
WINDOW_SIZE = 50
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Output paths (saved next to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.keras")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")


# ============================
# 1. Load datasets
# ============================
def load_datasets():
    """Load all CSV files and label them as working (1) or not-working (0)."""

    working_files = {
        os.path.join(BASE_DIR, "Painting_fixed_augmented.csv"): 1,
        os.path.join(BASE_DIR, "Sawing_fixed_augmented.csv"): 1,
        os.path.join(BASE_DIR, "Screwing_fixed_augmented.csv"): 1,
        os.path.join(BASE_DIR, "Lifting load_fixed_augmented.csv"): 1,
        os.path.join(BASE_DIR, "Masonry_fixed_augmented.csv"): 1,
        os.path.join(BASE_DIR, "Hammering_fixed_augmented.csv"): 1,
    }

    not_working_files = {
        os.path.join(BASE_DIR, "Sitting_fixed_augmented.csv"): 0,
        os.path.join(BASE_DIR, "Standing_fixed_augmented.csv"): 0,
    }

    dfs = []

    for file, label in {**working_files, **not_working_files}.items():
        if not os.path.exists(file):
            print(f"WARNING: File not found — {file}")
            continue
        df = pd.read_csv(file)
        df["binary_label"] = label
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No CSV data files found in {BASE_DIR}. "
            "Please ensure the training data is available."
        )

    data = pd.concat(dfs, ignore_index=True)
    print(f"Total samples loaded: {len(data)}")
    return data


# ============================
# 2. Create sliding windows
# ============================
def create_windows(X, y, window_size=WINDOW_SIZE):
    """Convert raw samples into overlapping windows of fixed size."""
    X_windows, y_windows = [], []
    for i in range(len(X) - window_size):
        X_windows.append(X[i : i + window_size])
        y_windows.append(y[i + window_size])
    return np.array(X_windows), np.array(y_windows)


# ============================
# 3. Build the CNN model
# ============================
def build_model(window_size=WINDOW_SIZE, n_features=3):
    """Build and compile a 1D CNN for binary classification."""
    model = Sequential([
        Input(shape=(window_size, n_features)),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ============================
# Main training pipeline
# ============================
def main():
    # --- Load data ---
    data = load_datasets()
    X_raw = data[["xacc", "yacc", "zacc"]].values
    y_raw = data["binary_label"].values
    print(f"Raw shape: X={X_raw.shape}, y={y_raw.shape}")

    # --- Create windows ---
    X, y = create_windows(X_raw, y_raw, WINDOW_SIZE)
    print(f"Windowed shape: X={X.shape}, y={y.shape}")

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --- Normalise with StandardScaler ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)

    # Save scaler for inference
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # --- Build & train model ---
    model = build_model()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
    )

    # --- Evaluate ---
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # --- Save model ---
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
