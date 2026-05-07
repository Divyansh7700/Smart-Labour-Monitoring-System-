"""
model.py — Model loading and prediction module.

Provides:
  - predict_output(input_data)  : predict from a full window (50×3 array)
  - predict_single(xacc, yacc, zacc) : predict from a single sample (demo)

The trained model (model.keras) and scaler (scaler.pkl) are loaded
once when this module is first imported.

Configuration:
  MODEL_DEBUG env var  — set to "1" to print prediction debug info
"""

import os
import numpy as np
import joblib

# ============================
# Constants
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.keras")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")
WINDOW_SIZE = 50
NUM_FEATURES = 3        # [xacc, yacc, zacc] — must match training
THRESHOLD = 0.5         # sigmoid threshold for binary classification

DEBUG_MODE = os.environ.get("MODEL_DEBUG", "0") == "1"

# ============================
# Load model & scaler at import time
# ============================
_model = None
_scaler = None
_load_error = None       # Stored so Flask can surface the error in the UI


def _load_resources():
    """Load the Keras model and scikit-learn scaler from disk."""
    global _model, _scaler, _load_error

    if not os.path.exists(MODEL_PATH):
        _load_error = (
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please run 'python train.py' first to train and save the model."
        )
        raise FileNotFoundError(_load_error)

    if not os.path.exists(SCALER_PATH):
        _load_error = (
            f"Scaler not found at '{SCALER_PATH}'. "
            "Please run 'python train.py' first to save the scaler."
        )
        raise FileNotFoundError(_load_error)

    # Import tensorflow only when needed (keeps startup fast if model is missing)
    from tensorflow.keras.models import load_model

    _model = load_model(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _load_error = None
    print(f"[model.py] Model loaded from {MODEL_PATH}")
    print(f"[model.py] Scaler loaded from {SCALER_PATH}")


# Attempt to load immediately on import
try:
    _load_resources()
except FileNotFoundError as e:
    print(f"[model.py] WARNING: {e}")


def get_model_status() -> dict:
    """Return the current load status of the model and scaler."""
    return {
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
        "error": _load_error,
    }


# ============================
# Prediction functions
# ============================
def predict_output(input_data):
    """
    Run a prediction on a full window of accelerometer data.

    Args:
        input_data: array-like of shape (50, 3) — 50 time-steps of
                    [xacc, yacc, zacc] values.  Accepts list-of-lists
                    or numpy arrays.

    Returns:
        dict with keys:
            "label"      : str  — "Working" or "Not Working"
            "confidence" : float — model sigmoid output (0-1)

    Raises:
        RuntimeError: if model/scaler not loaded
        ValueError: if input shape is wrong
    """
    if _model is None or _scaler is None:
        raise RuntimeError(
            "Model or scaler not loaded. "
            "Ensure model.keras and scaler.pkl exist, then restart the app."
        )

    # Convert to numpy array and validate shape
    data = np.array(input_data, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected input with {NUM_FEATURES} columns [xacc, yacc, zacc], "
            f"got shape {data.shape}."
        )

    if data.shape[0] != WINDOW_SIZE:
        raise ValueError(
            f"Expected {WINDOW_SIZE} rows (time-steps), got {data.shape[0]}."
        )

    if DEBUG_MODE:
        print(f"[DEBUG model] Input shape: {data.shape}")
        print(f"[DEBUG model] Input sample (first row): {data[0]}")

    # Scale the data using the fitted scaler
    # Scaler was fit on (N, 3) during training, so reshape → scale → reshape back
    data_scaled = _scaler.transform(data)

    if DEBUG_MODE:
        print(f"[DEBUG model] Scaled sample (first row): {np.round(data_scaled[0], 4)}")

    # Reshape for the model: (1, window_size, 3)
    data_input = data_scaled.reshape(1, WINDOW_SIZE, NUM_FEATURES)

    # Run prediction
    raw_pred = float(_model.predict(data_input, verbose=0)[0][0])
    label = "Working" if raw_pred > THRESHOLD else "Not Working"

    if DEBUG_MODE:
        print(f"[DEBUG model] Raw prediction: {raw_pred:.6f} → {label}")

    return {
        "label": label,
    }


def predict_single(xacc, yacc, zacc):
    """
    Convenience function: predict from a single (xacc, yacc, zacc) sample.

    Since the model expects a window of 50 time-steps, this helper
    replicates the single sample 50 times to fill the window.
    This is suitable for quick demos; for real-time use, accumulate
    50 consecutive readings and call predict_output() directly.

    Args:
        xacc : float — X-axis acceleration
        yacc : float — Y-axis acceleration
        zacc : float — Z-axis acceleration

    Returns:
        dict  — same as predict_output()
    """
    # Build a window by repeating the single sample
    window = np.tile([xacc, yacc, zacc], (WINDOW_SIZE, 1))
    return predict_output(window)
