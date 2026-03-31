<div align="center">

# 🤖 GCBOT — Gamified Cleaning Bot

**An AI-powered autonomous trash-collecting robot with real-time web control, live video streaming, YOLOv10 waste detection, and a combined AI + weight-based scoring system.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Arduino](https://img.shields.io/badge/Arduino-Uno-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://arduino.cc)
[![YOLOv10](https://img.shields.io/badge/YOLOv10-Ultralytics-FF6600?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-4-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)](https://raspberrypi.org)
[![License](https://img.shields.io/badge/License-Kavin%20and%20Team-blueviolet?style=for-the-badge)](#-license)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Architecture](#-system-architecture)
- [Features](#-features)
- [Hardware](#-hardware)
- [Software Stack](#-software-stack)
- [AI Model — YOLOv10 Trash Detection](#-ai-model--yolov10-trash-detection)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Control Protocol](#-control-protocol)
- [Scoring System](#-scoring-system)
- [Web Control Interface](#-web-control-interface)
- [Contributing](#-contributing)

---

## 🌟 Overview

**GCBOT** is a remotely controlled garbage-collecting robot built for competitions and real-world waste management challenges. It combines:

- 🏎️ **A 4-wheel drive chassis** controlled via a motor driver
- 🦾 **A servo-powered arm & gripper** for picking up trash
- 📷 **A live camera stream** from an onboard Raspberry Pi
- 🧠 **YOLOv10 AI** running real-time trash detection on the laptop
- 🏆 **Combined AI + Weight scoring** — points are based on both what the AI detected AND how heavy the object is
- ⚖️ **HX711 load cell** measures object weight for score multiplier
- 🌐 **A browser-based control panel** accessible from any device on the network

The entire system communicates over TCP sockets — the laptop runs the Flask web server, AI inference, and score computation; the Raspberry Pi acts as a bridge between the laptop and the Arduino; and the Arduino directly controls all motors, servos, and the load cell.

---

## 🎬 Demo

### 🖥️ Web Control Interface (Desktop)

> The full-screen landscape control panel with live AI Detection feed, movement D-pad, gripper & arm controls, and the real-time score badge.

![GCBOT Desktop Control UI](Screenshot%202026-03-26%20213021.png)

---

### 📱 Mobile Interface — Landscape Mode

> The interface automatically adapts to mobile landscape mode. All controls are touch-optimized and response is near-instant thanks to TCP_NODELAY.

| Mobile View (Portrait) | Mobile Landscape with Feed |
|:---:|:---:|
| ![Mobile Portrait](WhatsApp%20Image%202026-03-17%20at%208.20.31%20PM.jpeg) | ![Mobile Landscape](WhatsApp%20Image%202026-03-26%20at%206.47.48%20PM.jpeg) |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   GCBOT — Gamified Cleaning Bot                  │
│                                                                  │
│  ┌────────────┐   WiFi/TCP   ┌──────────────┐   Serial/USB      │
│  │   LAPTOP   │◄────────────►│ Raspberry Pi │◄─────────────────►│
│  │            │              │              │                    │
│  │ Flask App  │              │ pi_control   │   ┌─────────────┐  │
│  │ (Port 5001)│              │ (Port 5000)  │   │   Arduino   │  │
│  │            │              │              │   │             │  │
│  │ YOLOv10    │◄── Video ───►│ Camera       │   │ Motor Driver│  │
│  │ Inference  │   (Port 9999)│ Stream Tx    │   │ Servos (x4) │  │
│  │            │              │              │   │ Load Cell   │  │
│  │ Web UI     │◄── Cmds ────►│ Serial Relay │   │ HX711       │  │
│  └────────────┘              └──────────────┘   └─────────────┘  │
│        ▲                                                          │
│        │  HTTP (Browser)                                          │
│  ┌─────┴──────┐                                                   │
│  │ Any Device │ (Phone / Tablet / PC on same network)            │
│  └────────────┘                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Communication Flow

```mermaid
sequenceDiagram
    participant Browser as 🌐 Browser
    participant Laptop as 💻 Laptop (Flask)
    participant Pi as 🍓 Raspberry Pi
    participant Arduino as ⚡ Arduino

    Browser->>Laptop: HTTP POST /cmd {"cmd": "F"}
    Laptop->>Pi: TCP "F\n" (Port 5000)
    Pi->>Arduino: Serial "F\n"
    Arduino-->>Arduino: forward()

    Browser->>Laptop: HTTP GET /video_feed
    Pi-->>Laptop: TCP Video Frame (Port 9999)
    Laptop-->>Laptop: YOLOv10 Inference → tracks last_detected_class
    Laptop-->>Browser: MJPEG Stream

    Note over Browser,Arduino: DROP OBJECT Flow (AI + Weight Scoring)
    Browser->>Laptop: POST /cmd {"cmd": "WEIGHTNOW"}
    Laptop->>Laptop: weighnow_pending = True
    Laptop->>Pi: TCP "WEIGHTNOW\n"
    Pi->>Arduino: Serial "WEIGHTNOW\n"
    Arduino-->>Arduino: HX711 read → diff = current - lastWeight
    Arduino-->>Pi: Serial "W:23.45\n"
    Pi-->>Laptop: TCP "W:23.45\n" (broadcast)
    Laptop-->>Laptop: compute_combined_score(23.45g)
    Note over Laptop: score = round(base × (1 + 23.45/100))
    Laptop-->>Browser: JSON {score: 12, event: {cls: "Bottle", pts: 12, weight: 23.5}}
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎮 **WASD / D-pad Control** | Move forward, backward, left, right with keyboard or touch |
| 🦾 **Arm Lift Control** | Dual servo arm with animated angle ring indicator |
| 🤏 **Gripper Control** | Dual servo gripper (open/close) with visual feedback |
| 🗑️ **Drop Object** | One-tap: releases gripper → waits 2s → weighs → combines AI class + weight → scores |
| 📷 **Live Camera Feed** | MJPEG stream from Pi camera at ~30 FPS |
| 🤖 **AI Detection Feed** | YOLOv10 trash overlay feed (host only) with toggle |
| 🏆 **AI + Weight Scoring** | Score = `base_pts × (1 + weight_g / 100)` — heavier + rarer = more points |
| ⚖️ **Load Cell Integration** | HX711 reads weight diff on command, laptop combines with AI class |
| 📱 **Mobile Responsive** | Landscape-optimized with portrait rotation hint |
| ⚡ **Low Latency** | TCP_NODELAY + no blocking delays = near-zero command lag |
| 🔄 **Auto-Reconnect** | All TCP connections auto-reconnect on drop |

---

## 🔧 Hardware

### Components List

| Component | Quantity | Purpose |
|---|---|---|
| Arduino Uno | 1 | Motor & servo control, load cell reading |
| Raspberry Pi 4 | 1 | Video streaming, TCP bridge to Arduino |
| L298N Motor Driver | 1 | 4WD motor control |
| DC Gear Motors (TT Motor) | 4 | Drive wheels |
| SG90 / MG90S Servo | 4 | 2× Arm lift + 2× Gripper |
| HX711 Load Cell Amplifier | 1 | Weight sensing for scoring |
| Load Cell (1kg) | 1 | Detects collected trash weight |
| Pi Camera Module | 1 | Live video stream |
| 3S LiPo / 18650 Pack | 1 | Power supply |
| Robot Chassis (4WD) | 1 | Physical body |

### Wiring — Arduino Pin Map

```
Arduino Uno
├── Pin  2  → L298N IN1  (Left Motor A)
├── Pin  3  → L298N IN2  (Left Motor B)
├── Pin  4  → L298N IN3  (Right Motor A)
├── Pin  7  → L298N IN4  (Right Motor B)
├── Pin  8  → L298N ENA  (Left Motor PWM)
├── Pin 12  → L298N ENB  (Right Motor PWM)
├── Pin  5  → Gripper Servo A (grabA)
├── Pin  6  → Gripper Servo B (grabB) [mirrored]
├── Pin  9  → Lift Servo A (liftA)
├── Pin 10  → Lift Servo B (liftB) [mirrored]
├── Pin A0  → HX711 SCK
└── Pin A1  → HX711 DT
```

---

## 💻 Software Stack

| Layer | Technology |
|---|---|
| **Web Framework** | Flask (Python) |
| **AI Inference** | Ultralytics YOLOv10 (`best.pt`) |
| **Computer Vision** | OpenCV (`cv2`) |
| **Video Protocol** | MJPEG over HTTP (multipart) |
| **Control Protocol** | Raw TCP sockets with TCP_NODELAY |
| **Serial** | PySerial (Pi ↔ Arduino) |
| **Frontend** | Vanilla HTML/CSS/JS with glassmorphism UI |
| **Fonts** | Google Inter |
| **Microcontroller** | Arduino C++ with Servo.h + HX711.h |
| **Pi Script** | Python threading + socket server |

---

## 🧠 AI Model — YOLOv10 Trash Detection

The onboard AI model is a **custom-trained YOLOv10** fine-tuned on a multi-class garbage detection dataset. It runs locally on the **laptop CPU** in a dedicated inference thread at 416×416 resolution.

### Detected Waste Classes (18 classes)

> Aluminium foil · Bottle · Bottle cap · Broken glass · Can · Carton · Cigarette · Cup · Lid · Other litter · Other plastic · Paper · Plastic bag/wrapper · Plastic container · Pop tab · Straw · Styrofoam piece · Unlabeled litter

### Training Performance Graphs

<table>
<tr>
<td><b>Precision-Recall Curve (Train)</b></td>
<td><b>Confusion Matrix (Train)</b></td>
</tr>
<tr>
<td><img src="Trash_detection_Yolov10_StreamLit/train_graphs/PR_curve.png" alt="PR Curve" width="420"/></td>
<td><img src="Trash_detection_Yolov10_StreamLit/train_graphs/confusion_matrix_normalized.png" alt="Confusion Matrix" width="420"/></td>
</tr>
<tr>
<td><b>F1-Confidence Curve</b></td>
<td><b>Precision Curve</b></td>
</tr>
<tr>
<td><img src="Trash_detection_Yolov10_StreamLit/train_graphs/F1_curve.png" alt="F1 Curve" width="420"/></td>
<td><img src="Trash_detection_Yolov10_StreamLit/train_graphs/P_curve.png" alt="Precision Curve" width="420"/></td>
</tr>
</table>

### Validation Performance Graphs

<table>
<tr>
<td><b>Precision-Recall Curve (Val)</b></td>
<td><b>Confusion Matrix (Val)</b></td>
</tr>
<tr>
<td><img src="Trash_detection_Yolov10_StreamLit/val_graphs/PR_curve_val.png" alt="PR Curve Val" width="420"/></td>
<td><img src="Trash_detection_Yolov10_StreamLit/val_graphs/confusion_matrix_normalized_val.png" alt="Confusion Matrix Val" width="420"/></td>
</tr>
</table>

### Inference Optimizations

- ✅ Frame resized to **416×416** before inference (3× speedup vs full-res)
- ✅ **Duplicate frame skipping** — hashes frame identity, skips if unchanged
- ✅ **PyTorch thread capping** — prevents CPU over-subscription
- ✅ **Confidence threshold 0.25** — filters weak detections
- ✅ **Annotated frame scaled back** to original resolution after inference

---

## 📁 Project Structure

```
GCBOT/
│
├── app.py                              # 💻 Laptop: Flask server, YOLOv10 inference, AI+weight scoring
├── pi_control.py                       # 🍓 Raspberry Pi: TCP server + Serial relay to Arduino
│
├── gcbot_arduino/
│   └── gcbot_arduino.ino              # ⚡ Arduino: motors, servos, load cell (weight only)
│
├── Trash_detection_Yolov10_StreamLit/
│   ├── best.pt                        # 🧠 Trained YOLOv10 model weights
│   ├── train_graphs/                  # 📊 Training performance graphs
│   │   ├── PR_curve.png
│   │   ├── F1_curve.png
│   │   ├── P_curve.png
│   │   ├── R_curve.png
│   │   └── confusion_matrix_normalized.png
│   ├── val_graphs/                    # 📊 Validation performance graphs
│   │   ├── PR_curve_val.png
│   │   ├── F1_curve_val.png
│   │   ├── P_curve_val.png
│   │   ├── R_curve_val.png
│   │   └── confusion_matrix_normalized_val.png
│   ├── train_yolov10_garbage_detection.ipynb  # 📓 Training notebook
│   └── requirements.txt
│
├── .gitignore
├── Screenshot 2026-03-26 213021.png   # 🖼️ Desktop UI screenshot
├── WhatsApp Image 2026-03-17 *.jpeg   # 🖼️ Hardware & mobile UI photos
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+ on the **Laptop**
- Python 3.x on the **Raspberry Pi**
- Arduino IDE for flashing the `.ino` sketch

### 1. Laptop Setup

```bash
# Clone the repository
git clone https://github.com/your-username/gcbot.git
cd gcbot

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# Install dependencies
pip install flask ultralytics opencv-python numpy torch
```

### 2. Raspberry Pi Setup

```bash
# Copy pi_control.py to the Raspberry Pi
scp pi_control.py pi@kavin.local:~/

# SSH into Pi and install dependencies
ssh pi@kavin.local
pip install pyserial
```

### 3. Arduino Setup

1. Open `gcbot_arduino/gcbot_arduino.ino` in **Arduino IDE**
2. Install required libraries:
   - **Servo** (built-in)
   - **HX711** by bogde
3. Flash to your Arduino Uno

> ⚠️ **Calibrate your load cell!** Replace `2280.0` in the sketch with your actual calibration factor. Run a calibration sketch with a known weight to find the correct value.

### 4. Configuration

In `app.py`, update the Pi's hostname or IP:

```python
PI_IP        = "kavin.local"   # or use the Pi's IP e.g. "192.168.1.42"
CONTROL_PORT = 5000
VIDEO_PORT   = 9999
```

In `pi_control.py`, update the serial port if needed:

```python
ARDUINO_PORT = "/dev/ttyUSB0"   # use /dev/ttyACM0 for some Arduinos
ARDUINO_BAUD = 9600
```

---

## 🚀 How to Run

### Step 1 — Flash the Arduino

Connect the Arduino to the Raspberry Pi via USB, then flash `gcbot_arduino.ino` from your development machine.

### Step 2 — Start the Pi Bridge

```bash
# On Raspberry Pi
python3 pi_control.py
```

The Pi will:
- Open the Arduino serial port
- Start listening for laptop connections on port `5000`
- Start a video stream sender on port `9999` *(handled by the Pi camera script —  ensure your Pi camera streaming script is also running)*

### Step 3 — Start the Laptop Server

```bash
# On the Laptop (must be on same WiFi as the Pi)
python app.py
```

### Step 4 — Open the Web UI

Navigate to:

```
http://localhost:5001          # on the laptop (AI Detection feed)
http://<laptop-ip>:5001        # on any phone/tablet on the same WiFi
```

---

## 📡 Control Protocol

Commands are sent as **newline-terminated ASCII strings** over TCP:

| Command | Description |
|---|---|
| `F` | Move forward |
| `B` | Move backward |
| `L` | Turn left |
| `R` | Turn right |
| `S` | Stop all motors |
| `U<angle>` | Set arm lift angle (0–180°), e.g. `U90` |
| `G<angle>` | Set gripper angle (0–180°), e.g. `G120` |
| `WEIGHTNOW` | Trigger a weight check on the load cell |

### HTTP API (Flask)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve the control UI |
| `/cmd` | POST | Send a command to the robot |
| `/video_feed` | GET | Raw MJPEG camera stream |
| `/video_feed_detected` | GET | YOLOv10 annotated MJPEG stream |
| `/score` | GET | Current score + last scoring event breakdown |
| `/scorechart` | GET | Full score classification chart as JSON |

**Example cURL:**

```bash
# Send a movement command
curl -X POST http://localhost:5001/cmd \
     -H "Content-Type: application/json" \
     -d '{"cmd": "F"}'

# Get the score chart
curl http://localhost:5001/scorechart
```

---

## 🏆 Scoring System

Scoring uses a **combined AI detection + weight measurement** system. The laptop computes the final score using both what the AI sees and how heavy the object is.

### Formula

```
final_score = round(base_score × (1 + weight_g / 100))
```

- **`base_score`** — determined by the detected trash class (see chart below)
- **`weight_g`** — weight difference measured by the HX711 load cell (in grams)
- **No detection fallback** — if no trash was detected by AI: `base = max(1, weight_g ÷ 10)`

### Score Classification Chart

| Object Class | Base Points | @ 50g | @ 100g | @ 200g | Reason |
|---|---|---|---|---|---|
| Broken glass | 15 | 22 | 30 | 45 | Dangerous — bonus reward |
| Bottle | 10 | 15 | 20 | 30 | Recyclable |
| Can | 10 | 15 | 20 | 30 | Recyclable |
| Plastic bag / wrapper | 10 | 15 | 20 | 30 | High environmental impact |
| Carton | 8 | 12 | 16 | 24 | Recyclable |
| Cup | 8 | 12 | 16 | 24 | Disposable |
| Plastic container | 8 | 12 | 16 | 24 | Recyclable |
| Styrofoam piece | 8 | 12 | 16 | 24 | Hard to recycle |
| Other plastic | 7 | 10 | 14 | 21 | — |
| Aluminium foil | 5 | 7 | 10 | 15 | Recyclable |
| Paper | 5 | 7 | 10 | 15 | Recyclable |
| Straw | 5 | 7 | 10 | 15 | Environmental hazard |
| Other litter | 5 | 7 | 10 | 15 | — |
| Cigarette | 3 | 4 | 6 | 9 | Small but toxic |
| Lid | 3 | 4 | 6 | 9 | Small |
| Bottle cap | 3 | 4 | 6 | 9 | Small |
| Pop tab | 3 | 4 | 6 | 9 | Small |
| Unlabeled litter | 2 | 3 | 4 | 6 | Unknown fallback |
| **(No detection)** | weight÷10 | 5 | 10 | 20 | Weight-only fallback |

### Scoring Flow

```
1. User presses 🗑️ DROP OBJECT button in the browser
2. Gripper opens to 50° (releases trash)
3. After 2s delay, WEIGHTNOW command sent
4. Arduino reads HX711 load cell (avg 3 readings)
5. Arduino sends weight diff: "W:23.45\n"
6. Pi broadcasts W:23.45 to all connected laptops
7. Laptop's app.py combines:
   • last_detected_class (from YOLO inference) → base = 10 (Bottle)
   • weight_g = 23.45g → multiplier = 1.23
   • final_score = round(10 × 1.23) = 12 pts
8. Browser shows: "+12 Bottle (23.5g)" and updates the score badge
```

> 💡 **Tip:** The weight multiplier means heavier objects of the same class score higher. A full 200g bottle scores 30 pts vs. a small 50g bottle at 15 pts.

> ⚖️ **No detection fallback:** If the AI didn't detect the object before drop, scoring falls back to weight-only: 1 point per 10 grams.

---

## 🌐 Web Control Interface

The control UI is a **single-page, glassmorphism-styled dashboard** embedded in Flask (`app.py`). No external CSS framework is used.

### UI Features

- 🎮 **D-pad** for movement (touch + WASD keyboard)
- 🔵 **SVG ring indicator** for real-time arm/gripper angle display
- 🟢 **Score badge** (top-right, always visible) with scale-bounce animation on score update
- 📡 **Feed toggle** — switch between `Raw` and `AI Detection` video feed
- 🟢 **Status dot** with pulse animation indicating live connection
- ↻ **Portrait warning** — prompts rotation on mobile portrait mode
- ✚ **Crosshair overlay** on the video feed for targeting

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

```
Copyright © 2026 Kavin and Team. All Rights Reserved.

This project and all its source code, assets, trained models, and documentation
are the intellectual property of Kavin and Team. No part of this project may be
reproduced, distributed, modified, or used in any form without the express written
permission of the owners.

For licensing inquiries or collaboration, please open an issue or contact via GitHub.
```

---

<div align="center">

**Built with ❤️ for making the world cleaner, one piece of trash at a time 🌍♻️**

*GCBOT — Gamified Cleaning Bot*

</div>
