"""
sensor.py — Arduino serial sensor reader (MPU6050) with signal processing.

This module reads accelerometer (+ optional gyroscope) data from an Arduino
MPU6050 via pyserial and applies a multi-stage processing pipeline to
stabilise the signal before it reaches the ML model:

    Raw data  →  Baseline correction  →  Threshold filtering  →  Moving average  →  Output

Why each stage is needed:
  • Baseline correction:  The MPU6050 has a static offset (drift) that varies
    per sensor unit.  By capturing readings at rest on startup and subtracting
    the mean, we remove this constant bias.
  • Threshold filtering:  Even after calibration, tiny fluctuations remain when
    the sensor is stationary (< ±30–50 raw units).  Treating values below the
    threshold as zero eliminates this noise.
  • Moving average:  A sliding-window average of the last N readings smooths
    out any remaining high-frequency jitter, producing a clean signal.

Configuration via environment variables:
  SENSOR_PORT      — Serial port (default: COM7)
  SENSOR_BAUD      — Baud rate (default: 9600)
  SENSOR_DEBUG     — Set to "1" to enable debug printing
  SENSOR_THRESHOLD — Noise threshold (default: 40.0)
  SENSOR_SMOOTH    — Moving-average window size (default: 7)

Usage:
    from sensor import start_sensor, stop_sensor, get_sensor_data

    start_sensor(port="COM7", baud=9600)
    data = get_sensor_data()
    stop_sensor()
"""

import os
import threading
import time
from collections import deque

import numpy as np
import serial

# ============================
# Configuration (env-overridable)
# ============================
WINDOW_SIZE = 50                                                     # Model input window (50 time-steps)
DEFAULT_PORT = os.environ.get("SENSOR_PORT", "COM7")                 # Serial port
DEFAULT_BAUD = int(os.environ.get("SENSOR_BAUD", "9600"))            # Baud rate

CALIBRATION_SAMPLES = 30                                             # Baseline calibration samples
SMOOTHING_WINDOW = int(os.environ.get("SENSOR_SMOOTH", "7"))         # Moving-average window
NOISE_THRESHOLD = float(os.environ.get("SENSOR_THRESHOLD", "40.0"))  # Dead-zone threshold

DEBUG_MODE = os.environ.get("SENSOR_DEBUG", "0") == "1"              # Debug flag

# Reconnection backoff
_RECONNECT_BASE = 2    # seconds
_RECONNECT_MAX = 10    # seconds


# ============================
# Module-level shared state
# ============================
_lock = threading.Lock()                                 # Protects all shared state
_buffer = deque(maxlen=WINDOW_SIZE)                      # Sliding window fed to model
_latest_raw = None                                       # Most recent RAW accel reading [ax, ay, az]
_latest_processed = None                                 # Most recent PROCESSED accel reading
_latest_gyro = None                                      # Most recent gyroscope reading [gx, gy, gz] (optional)
_connected = False                                       # Serial connection status
_running = False                                         # Reader thread active flag
_serial_conn = None                                      # pyserial connection object
_thread = None                                           # Background reader thread
_error_msg = ""                                          # Last error (displayed in UI)

# --- Calibration state ---
_baseline = np.zeros(3)                                  # Offset [ax, ay, az] from calibration
_calibrated = False                                      # Has calibration completed?
_calibration_buf = []                                    # Temporary list during calibration

# --- Moving-average state ---
_smooth_buf = deque(maxlen=SMOOTHING_WINDOW)              # Recent readings for smoothing


# ============================
# Signal processing helpers
# ============================
def _apply_baseline(values: np.ndarray) -> np.ndarray:
    """
    Stage 1 — Baseline correction.
    Subtract the static offset captured during calibration so that
    a sensor at rest reads approximately [0, 0, 0].
    """
    return values - _baseline


def _apply_threshold(values: np.ndarray, threshold: float = NOISE_THRESHOLD) -> np.ndarray:
    """
    Stage 2 — Threshold (dead-zone) filtering.
    Any axis value whose absolute magnitude is below *threshold* is
    forced to zero.  This removes tiny fluctuations when the sensor
    is stationary.
    """
    result = values.copy()
    result[np.abs(result) < threshold] = 0.0
    return result


def _apply_moving_average(values: np.ndarray) -> np.ndarray:
    """
    Stage 3 — Moving-average smoothing.
    Average the last N readings (stored in _smooth_buf) to dampen
    high-frequency noise.  The new reading is appended first.
    """
    _smooth_buf.append(values.copy())
    if len(_smooth_buf) == 0:
        return values
    return np.mean(np.array(_smooth_buf), axis=0)


def _process(raw: np.ndarray) -> np.ndarray:
    """
    Full processing pipeline:
      raw  →  baseline correction  →  threshold filter  →  moving average
    """
    step1 = _apply_baseline(raw)
    step2 = _apply_threshold(step1)
    step3 = _apply_moving_average(step2)

    if DEBUG_MODE:
        print(f"[DEBUG sensor] raw={np.round(raw, 2)}  "
              f"calib={np.round(step1, 2)}  "
              f"thresh={np.round(step2, 2)}  "
              f"smooth={np.round(step3, 2)}")

    return step3


# ============================
# Calibration
# ============================
def _run_calibration():
    """
    Collect CALIBRATION_SAMPLES readings and compute the mean.
    This mean becomes the baseline offset that is subtracted from
    every future reading.

    Called automatically by the reader thread once enough samples
    have been collected after connection.
    """
    global _baseline, _calibrated, _calibration_buf

    if len(_calibration_buf) < CALIBRATION_SAMPLES:
        return  # Not enough samples yet

    data = np.array(_calibration_buf[:CALIBRATION_SAMPLES])
    _baseline = np.mean(data, axis=0)
    _calibrated = True
    _calibration_buf = []  # Free memory

    print(f"[sensor.py] Calibration complete — baseline offset: {np.round(_baseline, 2)}")


# ============================
# Serial reading helpers
# ============================
def _parse_line(line: str):
    """
    Parse a comma-separated line from the Arduino.

    Supported formats:
        ax,ay,az            → 3 values (accelerometer only)
        ax,ay,az,gx,gy,gz   → 6 values (accelerometer + gyroscope)

    Returns:
        tuple: (accel: np.ndarray[3], gyro: np.ndarray[3] or None)
        Returns (None, None) if the line is malformed.
    """
    parts = line.split(",")
    if len(parts) < 3:
        return None, None
    try:
        accel = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        gyro = None
        if len(parts) >= 6:
            gyro = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
        return accel, gyro
    except (ValueError, IndexError):
        return None, None


def _safe_close():
    """Close the serial connection if open."""
    global _serial_conn
    try:
        if _serial_conn and _serial_conn.is_open:
            _serial_conn.close()
    except Exception:
        pass
    _serial_conn = None


# ============================
# Background reader thread
# ============================
def _reader_loop(port: str, baud: int):
    """
    Continuously read lines from the serial port, run them through
    the processing pipeline, and append to the model-input buffer.

    Includes exponential backoff for reconnection attempts.
    """
    global _serial_conn, _connected, _running
    global _latest_raw, _latest_processed, _latest_gyro, _error_msg
    global _calibrated, _calibration_buf

    reconnect_delay = _RECONNECT_BASE

    while _running:
        # --- Connect (or reconnect) ---
        if _serial_conn is None or not _serial_conn.is_open:
            try:
                _serial_conn = serial.Serial(port, baud, timeout=1)
                time.sleep(2)  # Allow Arduino to reset after serial open
                with _lock:
                    _connected = True
                    _error_msg = ""
                    # Reset calibration state on each new connection
                    _calibrated = False
                    _calibration_buf = []
                    _smooth_buf.clear()
                    _buffer.clear()
                    _latest_gyro = None
                reconnect_delay = _RECONNECT_BASE  # Reset backoff on success
                print(f"[sensor.py] Connected to {port} @ {baud} baud — calibrating…")
            except serial.SerialException as e:
                with _lock:
                    _connected = False
                    _error_msg = f"Cannot open {port}: {e}"
                if DEBUG_MODE:
                    print(f"[DEBUG sensor] Connection failed, retry in {reconnect_delay}s: {e}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX)
                continue

        # --- Read one line ---
        try:
            raw_line = _serial_conn.readline()
            if not raw_line:
                continue  # Timeout — no data available

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            accel, gyro = _parse_line(line)
            if accel is None:
                if DEBUG_MODE:
                    print(f"[DEBUG sensor] Malformed line: {line!r}")
                continue  # Malformed data

            # Store gyroscope values for UI display (not used by model)
            if gyro is not None:
                with _lock:
                    _latest_gyro = gyro.tolist()

            # ---------- Calibration phase ----------
            if not _calibrated:
                _calibration_buf.append(accel.copy())
                with _lock:
                    _latest_raw = accel.tolist()
                    _error_msg = (
                        f"Calibrating… {len(_calibration_buf)}/{CALIBRATION_SAMPLES}"
                    )
                _run_calibration()
                continue  # Don't feed uncalibrated data to the model

            # ---------- Processing pipeline ----------
            processed = _process(accel)

            with _lock:
                _latest_raw = accel.tolist()
                _latest_processed = processed.tolist()
                _buffer.append(processed.tolist())
                _error_msg = ""

        except (ValueError, UnicodeDecodeError):
            pass  # Malformed line — skip
        except serial.SerialException as e:
            with _lock:
                _connected = False
                _error_msg = f"Serial read error: {e}"
            _safe_close()
            if DEBUG_MODE:
                print(f"[DEBUG sensor] Serial error, reconnecting in {reconnect_delay}s: {e}")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX)


# ============================
# Public API
# ============================
def start_sensor(port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD):
    """
    Start the background serial reader thread.
    Safe to call multiple times — only the first call has effect.
    Calibration happens automatically once the sensor is connected.
    """
    global _running, _thread

    if _running:
        return

    _running = True
    _thread = threading.Thread(
        target=_reader_loop,
        args=(port, baud),
        daemon=True,
        name="SensorReader",
    )
    _thread.start()
    print(f"[sensor.py] Sensor reader started (port={port}, baud={baud})")


def stop_sensor():
    """Stop the background reader and close the serial port."""
    global _running, _connected

    _running = False
    if _thread and _thread.is_alive():
        _thread.join(timeout=5)

    _safe_close()
    with _lock:
        _connected = False
    print("[sensor.py] Sensor reader stopped")


def get_sensor_data() -> dict:
    """
    Get the current sensor state — thread-safe.

    All values returned under "latest" and "window" are **fully processed**
    (baseline-corrected → threshold-filtered → moving-average smoothed).

    Returns:
        dict with keys:
            "connected"    : bool   — serial port connected?
            "calibrated"   : bool   — has baseline calibration finished?
            "buffer_ready" : bool   — does the buffer have 50 processed samples?
            "buffer_count" : int    — number of processed samples in buffer
            "window"       : list   — [[ax,ay,az], …]  (up to 50, for model)
            "latest"       : list | None — most recent processed [ax, ay, az]
            "latest_raw"   : list | None — most recent raw (unprocessed) reading
            "latest_gyro"  : list | None — most recent gyroscope [gx, gy, gz]
            "baseline"     : list   — current baseline offset [ax, ay, az]
            "error"        : str    — last error / status message
    """
    with _lock:
        return {
            "connected": _connected,
            "calibrated": _calibrated,
            "buffer_ready": len(_buffer) >= WINDOW_SIZE,
            "buffer_count": len(_buffer),
            "window": list(_buffer),
            "latest": list(_latest_processed) if _latest_processed else None,
            "latest_raw": list(_latest_raw) if _latest_raw else None,
            "latest_gyro": list(_latest_gyro) if _latest_gyro else None,
            "baseline": _baseline.tolist(),
            "error": _error_msg,
        }


def calibrate_sensor():
    """
    Force a re-calibration.
    Clears the current baseline and buffer so that the next
    CALIBRATION_SAMPLES readings are used to compute a new offset.
    Useful if the sensor has been moved to a new orientation.
    """
    global _calibrated, _calibration_buf, _baseline

    with _lock:
        _calibrated = False
        _calibration_buf = []
        _baseline = np.zeros(3)
        _smooth_buf.clear()
        _buffer.clear()

    print("[sensor.py] Re-calibration requested — collecting new baseline…")
