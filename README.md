# 👷 Smart Labour Monitoring System

A real-time worker activity classification system that uses an **MPU6050 accelerometer sensor** connected to an **Arduino** and a **1D Convolutional Neural Network (CNN)** to predict whether a labourer is **Working** or **Not Working**.

Built as a capstone project to demonstrate IoT-based activity recognition with deep learning.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)

---

## ✨ Features

- **Real-time Monitoring** — Live accelerometer data streaming from MPU6050 via Arduino serial connection
- **Deep Learning Classification** — 1D CNN model classifies activities as "Working" or "Not Working"
- **Signal Processing Pipeline** — Baseline correction → dead-zone filtering → moving-average smoothing
- **Auto-Calibration** — Automatically calibrates sensor baseline on startup
- **Manual Prediction** — Enter acceleration values manually to test the model
- **Live Web Dashboard** — Beautiful, responsive UI with real-time sensor values, connection status, and predictions
- **Debug Mode** — Built-in debug panel for diagnostics

---

## 🏗 System Architecture

```
┌──────────────┐    Serial    ┌────────────────┐    HTTP     ┌──────────────┐
│  MPU6050     │───────────►  │  Flask Server  │ ◄────────►  │  Web Browser │
│  + Arduino   │   (COM Port) │  (Python)      │  (API)      │  (Dashboard) │
└──────────────┘              └────────────────┘              └──────────────┘
                                     │
                              ┌──────┴──────┐
                              │             │
                         ┌────▼────┐  ┌─────▼─────┐
                         │ sensor  │  │   model   │
                         │  .py    │  │   .py     │
                         │         │  │           │
                         │ Signal  │  │ Keras CNN │
                         │ Process │  │ Predict   │
                         └─────────┘  └───────────┘
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Sensor** | MPU6050 (6-axis accelerometer + gyroscope) |
| **Microcontroller** | Arduino |
| **Backend** | Python, Flask |
| **Deep Learning** | TensorFlow / Keras (1D CNN) |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Serial Communication** | PySerial |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 📁 Project Structure

```
Smart-Labour-Monitoring-System/
├── app.py                  # Flask web server with API endpoints
├── model.py                # Model loading and prediction logic
├── sensor.py               # Serial sensor reader with signal processing
├── train.py                # Training script for the CNN model
├── model.keras             # Pre-trained Keras CNN model
├── scaler.pkl              # Fitted StandardScaler for feature normalization
├── requirements.txt        # Python dependencies
├── SMARTLABOURMONITORINGSYSTEM.ipynb  # Jupyter notebook (exploration/training)
├── templates/
│   └── index.html          # Web dashboard (Jinja2 template)
├── static/
│   └── style.css           # Dashboard styles
└── .gitignore
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Arduino with MPU6050 sensor (for live monitoring)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Divyansh7700/Smart-Labour-Monitoring-System-.git
   cd Smart-Labour-Monitoring-System-
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 💡 Usage

### Live Monitoring (with Arduino + MPU6050)

1. Connect the Arduino with the MPU6050 sensor to your computer via USB
2. Start the application: `python app.py`
3. Open `http://localhost:5000` in your browser
4. Click **"Start Live"** to begin real-time monitoring
5. The system will auto-calibrate, then start predicting activity

### Manual Prediction (without hardware)

1. Start the application: `python app.py`
2. Open `http://localhost:5000`
3. Scroll to the **Manual Input** section
4. Enter X, Y, Z acceleration values
5. Click **"Predict Activity"** to get the classification result

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| **Architecture** | 1D CNN (Conv1D) |
| **Input Shape** | (50, 3) — 50 time-steps × 3 axes |
| **Layers** | Conv1D(64) → MaxPool → Conv1D(128) → MaxPool → Flatten → Dense(64) → Dropout(0.5) → Dense(1, sigmoid) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss** | Binary Crossentropy |
| **Output** | Binary — "Working" (>0.5) / "Not Working" (≤0.5) |

### Training Activities

| Working (Label = 1) | Not Working (Label = 0) |
|---------------------|------------------------|
| Painting | Sitting |
| Sawing | Standing |
| Screwing | |
| Lifting Load | |
| Masonry | |
| Hammering | |

### Retraining the Model

```bash
python train.py
```

This will load CSV datasets, create sliding windows, train the CNN, and save `model.keras` + `scaler.pkl`.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard |
| `POST` | `/predict` | Manual prediction (form data: xacc, yacc, zacc) |
| `GET` | `/live-data` | JSON: live sensor values + model prediction |
| `GET` | `/sensor-status` | JSON: sensor connection status |
| `POST` | `/recalibrate` | Trigger sensor re-calibration |
| `GET` | `/debug` | JSON: full debug information |

---

## ⚙️ Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `SENSOR_PORT` | `COM7` | Arduino serial port |
| `SENSOR_BAUD` | `9600` | Serial baud rate |
| `APP_DEBUG` | `0` | Enable verbose Flask logging (`1` to enable) |
| `SENSOR_DEBUG` | `0` | Print sensor pipeline values (`1` to enable) |
| `MODEL_DEBUG` | `0` | Print model prediction internals (`1` to enable) |

---

## 📄 License

This project is developed as a Capstone Project.

---

<p align="center">Made with ❤️ as a Capstone Project</p>
