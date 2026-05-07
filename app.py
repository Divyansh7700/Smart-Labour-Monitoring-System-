"""
app.py — Flask web application for Labour Activity Monitoring.

Routes:
    GET  /              → Renders the homepage with input form + live monitor
    POST /predict       → Accepts accelerometer values (manual), returns prediction
    GET  /live-data     → JSON: live sensor values + model prediction
    GET  /sensor-status → JSON: serial connection status
    POST /recalibrate   → Triggers sensor re-calibration
    GET  /debug         → JSON: full debug information

Configuration (environment variables):
    SENSOR_PORT   — Arduino serial port (default: COM7)
    SENSOR_BAUD   — Serial baud rate (default: 9600)
    APP_DEBUG     — "1" to enable Flask debug mode with verbose logging
    SENSOR_DEBUG  — "1" to print sensor pipeline values
    MODEL_DEBUG   — "1" to print model prediction internals

Usage:
    python app.py
"""

import os
from flask import Flask, render_template, request, jsonify

# ============================
# Create Flask app
# ============================
app = Flask(__name__)

APP_DEBUG = os.environ.get("APP_DEBUG", "0") == "1"

# ============================
# Start the background sensor reader on app startup
# ============================
SENSOR_PORT = os.environ.get("SENSOR_PORT", "COM7")
SENSOR_BAUD = int(os.environ.get("SENSOR_BAUD", "9600"))

from sensor import start_sensor  # noqa: E402
start_sensor(port=SENSOR_PORT, baud=SENSOR_BAUD)


# ============================
# Routes
# ============================
@app.route("/")
def home():
    """Render the homepage with the prediction form and live monitor."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle manual prediction requests.
    Reads xacc, yacc, zacc from the form, calls the model,
    and returns the result to the same page.
    """
    try:
        # --- Validate and parse form inputs ---
        xacc_str = request.form.get("xacc", "").strip()
        yacc_str = request.form.get("yacc", "").strip()
        zacc_str = request.form.get("zacc", "").strip()

        if not xacc_str or not yacc_str or not zacc_str:
            return render_template(
                "index.html",
                error="All three fields (X, Y, Z acceleration) are required.",
            )

        try:
            xacc = float(xacc_str)
            yacc = float(yacc_str)
            zacc = float(zacc_str)
        except ValueError:
            return render_template(
                "index.html",
                error="Please enter valid numeric values for all acceleration fields.",
            )

        # --- Run prediction ---
        from model import predict_single

        result = predict_single(xacc, yacc, zacc)

        return render_template(
            "index.html",
            prediction=result["label"],
            xacc=xacc_str,
            yacc=yacc_str,
            zacc=zacc_str,
        )

    except FileNotFoundError as e:
        return render_template(
            "index.html",
            error=f"Model not found: {e}. Please run 'python train.py' first.",
        )
    except RuntimeError as e:
        return render_template(
            "index.html",
            error=f"Model error: {e}",
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"An unexpected error occurred: {e}",
        )


# ============================
# Live data API endpoints
# ============================
@app.route("/live-data")
def live_data():
    """
    JSON endpoint: returns current sensor values + model prediction.
    Called by the frontend JavaScript every ~1 second.

    Response format:
    {
        "connected": true/false,
        "calibrated": true/false,
        "buffer_ready": true/false,
        "buffer_count": 0-50,
        "sensor_values": [ax, ay, az] or null,
        "sensor_values_raw": [ax, ay, az] or null,
        "gyro_values": [gx, gy, gz] or null,
        "baseline": [ax, ay, az],
        "prediction": "Working" / "Not Working" / null,
        "confidence": 0.0-1.0 or null,
        "error": "" or error message,
        "timestamp": ISO timestamp
    }
    """
    from sensor import get_sensor_data
    from datetime import datetime

    sensor = get_sensor_data()

    response = {
        "connected": sensor["connected"],
        "calibrated": sensor.get("calibrated", False),
        "buffer_ready": sensor["buffer_ready"],
        "buffer_count": sensor["buffer_count"],
        "sensor_values": sensor["latest"],
        "sensor_values_raw": sensor.get("latest_raw"),
        "gyro_values": sensor.get("latest_gyro"),
        "baseline": sensor.get("baseline", [0, 0, 0]),
        "prediction": None,
        "error": sensor["error"],
        "timestamp": datetime.now().isoformat(),
    }
    # Run prediction only when the buffer has 50 processed samples
    if sensor["buffer_ready"]:
        try:
            from model import predict_output

            result = predict_output(sensor["window"])
            response["prediction"] = result["label"]
        except Exception as e:
            response["error"] = f"Prediction error: {e}"

    return jsonify(response)


@app.route("/sensor-status")
def sensor_status():
    """JSON endpoint: quick status check (no prediction)."""
    from sensor import get_sensor_data

    sensor = get_sensor_data()
    return jsonify({
        "connected": sensor["connected"],
        "calibrated": sensor.get("calibrated", False),
        "buffer_count": sensor["buffer_count"],
        "buffer_ready": sensor["buffer_ready"],
        "error": sensor["error"],
    })


@app.route("/recalibrate", methods=["POST"])
def recalibrate():
    """
    Trigger a sensor re-calibration.
    Clears the baseline and buffer so a fresh offset is computed.
    """
    from sensor import calibrate_sensor

    calibrate_sensor()
    return jsonify({"status": "ok", "message": "Re-calibration started"})


@app.route("/debug")
def debug_info():
    """
    JSON endpoint: full debug information for diagnostics.
    Returns sensor state, model status, and configuration.
    """
    from sensor import get_sensor_data
    from model import get_model_status

    sensor = get_sensor_data()
    model_status = get_model_status()

    return jsonify({
        "sensor": {
            "connected": sensor["connected"],
            "calibrated": sensor.get("calibrated", False),
            "buffer_count": sensor["buffer_count"],
            "buffer_ready": sensor["buffer_ready"],
            "baseline": sensor.get("baseline", [0, 0, 0]),
            "latest_raw": sensor.get("latest_raw"),
            "latest_processed": sensor.get("latest"),
            "latest_gyro": sensor.get("latest_gyro"),
            "error": sensor["error"],
        },
        "model": model_status,
        "config": {
            "sensor_port": SENSOR_PORT,
            "sensor_baud": SENSOR_BAUD,
            "app_debug": APP_DEBUG,
        },
    })


# ============================
# Entry point
# ============================
if __name__ == "__main__":
    # use_reloader=False prevents the sensor thread from being started twice
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
