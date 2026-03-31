"""
GCBOT – Combined Control & Video Stream App
Merges streamrx.py (video + Roboflow inference) and laptop_control.py (Flask control)
into a single file with a modern dark UI.
"""

from flask import Flask, Response, render_template_string, request, jsonify
import socket, struct, threading, time, os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Efficiency: limit PyTorch to physical CPU cores (avoids thread over-subscription)
torch.set_num_threads(os.cpu_count() or 4)

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
PI_IP        = "kavin.local"          # Raspberry Pi hostname / IP
CONTROL_PORT = 5000
VIDEO_PORT   = 9999

# Local YOLOv10 model
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Trash_detection_Yolov10_StreamLit", "best.pt")
yolo_model = YOLO(_MODEL_PATH)   # loaded once at startup; thread-safe for inference

# Inference resolution — smaller = faster; 320 is fast, 416 is a good balance
INFER_SIZE = 416

# ═══════════════════════════════════════════════════════════
#  SCORING SYSTEM
# ═══════════════════════════════════════════════════════════
# Base points per detected trash class (TACO dataset labels)
SCORE_MAP = {
    "Broken glass":          15,   # dangerous — bonus reward
    "Bottle":                10,   # recyclable
    "Can":                   10,   # recyclable
    "Plastic bag - wrapper": 10,   # high environmental impact
    "Carton":                 8,   # recyclable
    "Cup":                    8,   # disposable
    "Plastic container":      8,   # recyclable
    "Styrofoam piece":        8,   # hard to recycle
    "Other plastic":          7,
    "Aluminium foil":         5,   # recyclable
    "Paper":                  5,   # recyclable
    "Other litter":           5,
    "Straw":                  5,   # environmental hazard
    "Cigarette":              3,   # small but toxic
    "Lid":                    3,
    "Bottle cap":             3,
    "Pop tab":                3,
    "Unlabeled litter":       2,
}

# Weight scoring constants
W_REF       = 100.0   # grams — at W_REF the multiplier doubles the base score
MIN_WEIGHT  =   5.0   # grams — below this, treat as no object (noise rejection)

# ═══════════════════════════════════════════════════════════
#  GLOBALS
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)

latest_frame   = None
detected_frame = None
running        = True
latest_score   = 0

# Last detection snapshot (updated every inference frame)
last_detected_class = None   # e.g. "Bottle"
last_detected_conf  = 0.0    # e.g. 0.87

# Set True when WEIGHTNOW is sent; cleared once W: reply arrives
weighnow_pending = False
# Info about the last scored event (for the UI)
last_score_event = {"pts": 0, "cls": "-", "weight": 0.0, "base": 0, "mult": 1.0}

pi_lock = threading.Lock()
pi_sock = None

# ═══════════════════════════════════════════════════════════
#  PI CONNECTION (control channel)
# ═══════════════════════════════════════════════════════════
def compute_combined_score(weight_g: float) -> dict:
    """
    Formula:  final = round(base × (1 + weight_g / W_REF))

    If no object was detected → weight-only fallback:
      base = max(1, round(weight_g / 10))   (1 pt per 10 g)

    Returns dict: {pts, cls, weight, base, mult}
    """
    global latest_score, last_score_event

    cls  = last_detected_class
    conf = last_detected_conf

    if cls and cls in SCORE_MAP:
        base = SCORE_MAP[cls]
    elif cls:
        base = 5                       # detected but class not in map
    else:
        # No detection — weight-only
        cls  = "(weight only — no detection)"
        conf = 0.0
        base = max(1, round(weight_g / 10))

    mult  = round(1.0 + weight_g / W_REF, 2)
    pts   = round(base * mult)
    latest_score += pts

    event = {"pts": pts, "cls": cls, "weight": round(weight_g, 1),
             "base": base, "mult": mult, "conf": round(conf, 2),
             "total": latest_score}
    last_score_event = event
    print(f"[SCORE] +{pts} pts | {cls} (conf={conf:.2f}) | {weight_g:.1f}g "
          f"| base={base} × {mult:.2f} → total={latest_score}")
    return event


def pi_reader(sock):
    """Read lines sent by the Pi and update state."""
    global weighnow_pending, latest_score
    buf = b""
    try:
        while True:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = line.decode(errors="ignore").strip()

                if msg.startswith("SCORE:"):
                    # Legacy path — keep for backwards-compat
                    try:
                        latest_score = int(msg.split(":", 1)[1])  # noqa: F811
                        print(f"[SCORE-legacy] {latest_score}")
                    except ValueError:
                        pass

                elif msg.startswith("W:"):
                    raw = msg[2:]   # strip "W:"
                    try:
                        diff_g = float(raw)
                        if weighnow_pending and diff_g >= MIN_WEIGHT:
                            weighnow_pending = False
                            compute_combined_score(diff_g)
                        elif weighnow_pending:
                            weighnow_pending = False
                            print(f"[SCORE] Skipped — weight {diff_g:.1f}g < MIN ({MIN_WEIGHT}g)")
                    except ValueError:
                        pass   # debug strings like ERR, baseline_reset, etc.
    except Exception as e:
        print(f"[READER] {e}")

def connect_to_pi():
    global pi_sock
    while True:
        try:
            print(f"[CTRL] Connecting to Pi at {PI_IP}:{CONTROL_PORT} …")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((PI_IP, CONTROL_PORT))
            s.settimeout(None)
            with pi_lock:
                pi_sock = s
            print("[CTRL] Connected ✓")
            # start reader for this socket; when it exits, reconnect
            pi_reader(s)
            print("[CTRL] Connection lost — reconnecting…")
        except Exception as e:
            print(f"[CTRL] Connection failed: {e}")
        finally:
            with pi_lock:
                if pi_sock is s:
                    pi_sock = None
        time.sleep(2)

def send_to_pi(cmd: str):
    global pi_sock
    with pi_lock:
        if pi_sock is None:
            return False, "Not connected"
        try:
            pi_sock.sendall((cmd + "\n").encode())
            return True, "sent"
        except Exception as e:
            print(f"[CTRL] Send error: {e}")
            try:
                pi_sock.close()
            except:
                pass
            pi_sock = None
            return False, str(e)

threading.Thread(target=connect_to_pi, daemon=True).start()

# ═══════════════════════════════════════════════════════════
#  VIDEO RECEIVER THREAD
# ═══════════════════════════════════════════════════════════
def video_receiver():
    global latest_frame, running

    while running:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            print(f"[VIDEO] Connecting to {PI_IP}:{VIDEO_PORT} …")
            sock.connect((PI_IP, VIDEO_PORT))
            sock.settimeout(None)
            print("[VIDEO] Connected ✓")
        except Exception as e:
            print(f"[VIDEO] Connection failed: {e}")
            time.sleep(2)
            continue

        data = b""
        payload_size = struct.calcsize(">L")

        try:
            while running:
                while len(data) < payload_size:
                    pkt = sock.recv(4096)
                    if not pkt:
                        raise ConnectionError("stream ended")
                    data += pkt

                msg_size = struct.unpack(">L", data[:payload_size])[0]
                data = data[payload_size:]

                while len(data) < msg_size:
                    data += sock.recv(4096)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = cv2.imdecode(
                    np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    latest_frame = frame
        except Exception as e:
            print(f"[VIDEO] Error: {e}")
        finally:
            sock.close()
            time.sleep(1)

threading.Thread(target=video_receiver, daemon=True).start()

# ═══════════════════════════════════════════════════════════
#  INFERENCE THREAD (local YOLOv10)
# ═══════════════════════════════════════════════════════════
def inference_loop():
    """Run local YOLOv10 inference as fast as the CPU allows.

    Optimisations applied:
      • Frame resized to INFER_SIZE before inference (3× speedup vs full-res)
      • imgsz= passed so YOLO doesn't resize again internally
      • Skip inference when the frame hasn't changed (avoids duplicate work)
      • torch thread count capped at startup to stop CPU over-subscription
    """
    global detected_frame, last_detected_class, last_detected_conf
    last_hash = None

    while running:
        frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue

        # ── Skip if frame hasn't changed since last inference
        fhash = id(frame)
        if fhash == last_hash:
            time.sleep(0.02)
            continue
        last_hash = fhash

        # ── Resize to inference resolution
        small = cv2.resize(frame, (INFER_SIZE, INFER_SIZE),
                           interpolation=cv2.INTER_LINEAR)

        try:
            results = yolo_model(small, imgsz=INFER_SIZE, conf=0.25, verbose=False)
            annotated = results[0].plot()
            detected_frame = cv2.resize(annotated, (frame.shape[1], frame.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

            # ── Track highest-confidence detection for scoring
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best_idx  = int(boxes.conf.argmax())
                best_cls  = results[0].names[int(boxes.cls[best_idx])]
                best_conf = float(boxes.conf[best_idx])
                last_detected_class = best_cls
                last_detected_conf  = best_conf
            # Note: we do NOT clear last_detected_class when nothing is detected
            # so the last seen object is still used when DROP is pressed
        except Exception as e:
            print(f"[INFER] {e}")

threading.Thread(target=inference_loop, daemon=True).start()

# ═══════════════════════════════════════════════════════════
#  MJPEG GENERATOR
# ═══════════════════════════════════════════════════════════
def gen_mjpeg():
    """Raw feed – sent to all remote clients."""
    while running:
        frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )
        time.sleep(0.033)  # ~30 fps cap

def gen_mjpeg_detected():
    """AI-overlay feed – served only to the host device."""
    while running:
        frame = detected_frame if detected_frame is not None else latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )
        time.sleep(0.033)

# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════
@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_detected")
def video_feed_detected():
    return Response(gen_mjpeg_detected(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cmd", methods=["POST"])
def cmd():
    global weighnow_pending
    data = request.get_json(silent=True)
    if not data or "cmd" not in data:
        return jsonify(ok=False, error="no cmd"), 400
    command = str(data["cmd"]).strip()
    if command == "WEIGHTNOW":
        weighnow_pending = True   # arm the weight-triggered scorer
    ok, info = send_to_pi(command)
    return jsonify(ok=ok, info=info)

@app.route("/score")
def score():
    return jsonify(score=latest_score, event=last_score_event)

@app.route("/scorechart")
def scorechart():
    rows = [(cls, pts) for cls, pts in sorted(SCORE_MAP.items(), key=lambda x: -x[1])]
    return jsonify(chart=rows, W_REF=W_REF, MIN_WEIGHT=MIN_WEIGHT)

# ═══════════════════════════════════════════════════════════
#  FRONTEND
# ═══════════════════════════════════════════════════════════
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="screen-orientation" content="landscape">
<meta name="mobile-web-app-capable" content="yes">
<title>GCBOT Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  /* ── Reset ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  /* ── Palette ── */
  :root {
    --bg:        #0b0e14;
    --surface:   rgba(255,255,255,0.04);
    --glass:     rgba(255,255,255,0.07);
    --border:    rgba(255,255,255,0.10);
    --text:      #e2e8f0;
    --muted:     #94a3b8;
    --accent:    #6366f1;
    --accent-g:  linear-gradient(135deg, #6366f1, #a78bfa);
    --green:     #22c55e;
    --red:       #ef4444;
    --orange:    #f59e0b;
    --blue:      #3b82f6;
    --purple:    #a855f7;
    --radius:    16px;
  }

  html, body {
    height: 100%;
    height: 100dvh;
    overflow: hidden;
  }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
    -webkit-tap-highlight-color: transparent;
    margin: 0;
    padding: 0;
  }

  /* Force landscape orientation hint */
  @media screen and (max-width: 900px) and (orientation: portrait) {
    body::before {
      content: '↻ Please rotate your device to landscape mode';
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: rgba(0,0,0,0.9);
      color: white;
      padding: 20px 30px;
      border-radius: 12px;
      z-index: 9999;
      font-size: 1.1rem;
      text-align: center;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    }
  }

  /* ── Header ── */
  .header {
    text-align: center;
    padding: 8px 0;
    flex-shrink: 0;
  }
  .header h1 {
    font-size: 1.1rem;
    font-weight: 700;
    background: var(--accent-g);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin: 0;
  }
  .header .sub {
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 2px;
  }
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.4; }
  }

  /* ── Stream Card ── */
  .stream-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
  }
  .stream-card img {
    width: 100%;
    flex: 1;
    min-height: 0;
    display: block;
    background: #111;
    object-fit: contain;
  }
  .stream-label {
    padding: 4px 12px;
    font-size: 0.65rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
    justify-content: space-between;
  }
  .feed-toggle {
    display: flex;
    gap: 0;
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px;
    flex-shrink: 0;
  }
  .feed-toggle button {
    border: none;
    background: transparent;
    color: var(--muted);
    font-size: 0.6rem;
    font-family: inherit;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 16px;
    cursor: pointer;
    transition: background 0.18s, color 0.18s;
    white-space: nowrap;
  }
  .feed-toggle button.active {
    background: var(--accent);
    color: #fff;
  }

  /* ── Controls Container ── */
  .controls {
    width: 100%;
    max-width: 1200px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
  }

  /* ── Landscape Layout ── */
  .landscape-container {
    display: flex;
    flex-direction: row;
    gap: 12px;
    width: 100%;
    padding: 8px 12px 8px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
  .landscape-left {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
  }
  .landscape-center {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  .landscape-right {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    justify-content: center;
  }
  /* ── Glass Panel ── */
  .panel {
    background: var(--glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .panel-title {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
  }

  .angle-display {
    font-size: 1.5rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1;
  }
  .angle-unit {
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--muted);
  }

  /* ── Buttons Row ── */
  .btn-row {
    display: flex;
    gap: 10px;
  }

  .ctrl-btn {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: 2px solid var(--border);
    background: var(--surface);
    color: #fff;
    font-size: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: transform 0.12s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
    position: relative;
    overflow: hidden;
  }
  .ctrl-btn::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    opacity: 0;
    transition: opacity 0.2s;
  }
  .ctrl-btn:active, .ctrl-btn.active {
    transform: scale(0.92);
  }

  /* grab */
  .ctrl-btn.grab        { border-color: rgba(34,197,94,0.4); }
  .ctrl-btn.grab:hover  { border-color: var(--green); box-shadow: 0 0 20px rgba(34,197,94,0.25); }
  .ctrl-btn.grab.active { border-color: var(--green); box-shadow: 0 0 28px rgba(34,197,94,0.4); background: rgba(34,197,94,0.12); }

  /* release */
  .ctrl-btn.release        { border-color: rgba(239,68,68,0.4); }
  .ctrl-btn.release:hover  { border-color: var(--red); box-shadow: 0 0 20px rgba(239,68,68,0.25); }
  .ctrl-btn.release.active { border-color: var(--red); box-shadow: 0 0 28px rgba(239,68,68,0.4); background: rgba(239,68,68,0.12); }

  /* lift */
  .ctrl-btn.lift        { border-color: rgba(59,130,246,0.4); }
  .ctrl-btn.lift:hover  { border-color: var(--blue); box-shadow: 0 0 20px rgba(59,130,246,0.25); }
  .ctrl-btn.lift.active { border-color: var(--blue); box-shadow: 0 0 28px rgba(59,130,246,0.4); background: rgba(59,130,246,0.12); }

  /* drop */
  .ctrl-btn.drop        { border-color: rgba(245,158,11,0.4); }
  .ctrl-btn.drop:hover  { border-color: var(--orange); box-shadow: 0 0 20px rgba(245,158,11,0.25); }
  .ctrl-btn.drop.active { border-color: var(--orange); box-shadow: 0 0 28px rgba(245,158,11,0.4); background: rgba(245,158,11,0.12); }

  /* movement */
  .ctrl-btn.move        { border-color: rgba(168,85,247,0.4); }
  .ctrl-btn.move:hover  { border-color: var(--purple); box-shadow: 0 0 20px rgba(168,85,247,0.25); }
  .ctrl-btn.move.active { border-color: var(--purple); box-shadow: 0 0 28px rgba(168,85,247,0.4); background: rgba(168,85,247,0.12); }

  /* drop-object */
  .drop-obj-btn {
    width: 100%;
    padding: 7px 0;
    border-radius: 10px;
    border: 2px solid rgba(249,115,22,0.5);
    background: rgba(249,115,22,0.08);
    color: #fb923c;
    font-size: 0.7rem;
    font-family: inherit;
    font-weight: 700;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, transform 0.1s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    margin-top: 4px;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
  }
  .drop-obj-btn:hover  { background: rgba(249,115,22,0.18); border-color: #fb923c; }
  .drop-obj-btn:active { transform: scale(0.96); }
  .drop-obj-btn.busy   { opacity: 0.5; pointer-events: none; }

  /* ── D-pad grid ── */
  .dpad {
    display: grid;
    grid-template-columns: 60px 44px 60px;
    grid-template-rows: 60px 44px 60px;
    gap: 5px;
    justify-items: center;
    align-items: center;
  }
  .dpad .fwd   { grid-column: 2; grid-row: 1; }
  .dpad .left  { grid-column: 1; grid-row: 2; }
  .dpad .stp   { grid-column: 2; grid-row: 2; }
  .dpad .right { grid-column: 3; grid-row: 2; }
  .dpad .back  { grid-column: 2; grid-row: 3; }

  /* ── Progress arc (SVG ring) ── */
  .ring-wrap {
    position: relative;
    width: 80px;
    height: 80px;
  }
  .ring-wrap svg {
    transform: rotate(-90deg);
    width: 80px;
    height: 80px;
  }
  .ring-bg {
    fill: none;
    stroke: rgba(255,255,255,0.06);
    stroke-width: 6;
  }
  .ring-fg {
    fill: none;
    stroke-width: 6;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.08s linear;
  }
  .ring-value {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  /* ── Footer ── */
  .footer {
    display: none;
  }

  /* ── Responsive Media Queries ── */
  
  /* Mobile Landscape Mode - Optimize for horizontal layout */
  @media (max-height: 600px) {
    body {
      padding: 0;
    }
    .header {
      padding: 3px 0;
    }
    .header h1 {
      font-size: 0.85rem;
    }
    .header .sub {
      font-size: 0.5rem;
    }
    .landscape-container {
      padding: 0 6px 6px 6px;
      gap: 6px;
    }
    .panel {
      padding: 8px;
      gap: 6px;
      border-radius: 8px;
    }
    .panel-title {
      font-size: 0.5rem;
    }
    .ctrl-btn {
      width: 45px;
      height: 45px;
    }
    .ctrl-btn.stop {
      width: 34px;
      height: 34px;
    }
    .dpad {
      grid-template-columns: 45px 34px 45px;
      grid-template-rows: 45px 34px 45px;
      gap: 3px;
    }
    .ring-wrap {
      width: 55px;
      height: 55px;
    }
    .ring-wrap svg {
      width: 55px;
      height: 55px;
    }
    .angle-display {
      font-size: 1rem;
    }
    .angle-unit {
      font-size: 0.55rem;
    }
    .btn-row {
      gap: 6px;
    }
    .stream-label {
      padding: 2px 6px;
      font-size: 0.5rem;
    }
    .stream-card {
      border-radius: 6px;
    }
  }

  /* ── Responsive: keep row layout always, just scale elements on smaller screens ── */
  @media (max-width: 700px) {
    .landscape-container { gap: 8px; padding: 4px 8px 8px; }
    .panel { padding: 10px 8px; gap: 8px; }
    .ctrl-btn { width: 50px; height: 50px; font-size: 1.1rem; }
    .ctrl-btn.stop { width: 36px; height: 36px; }
    .dpad {
      grid-template-columns: 50px 36px 50px;
      grid-template-rows: 50px 36px 50px;
      gap: 3px;
    }
    .ring-wrap { width: 65px; height: 65px; }
    .ring-wrap svg { width: 65px; height: 65px; }
    .angle-display { font-size: 1.1rem; }
    .btn-row { gap: 6px; }
  }
  /* ── Score badge ── */
  .score-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px 4px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text);
    backdrop-filter: blur(8px);
  }
  .score-badge .score-icon { font-size: 0.85rem; }
  .score-badge .score-val {
    font-size: 1rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    background: linear-gradient(135deg, #22c55e, #86efac);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    transition: transform 0.15s ease;
  }
  .score-badge .score-val.bump {
    transform: scale(1.35);
  }
</style>
</head>
<body>

<!-- Fixed score badge (always visible, top-right) -->
<div style="position:fixed;top:10px;right:12px;z-index:999;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:6px 16px 8px;text-align:center;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px)">
  <div style="font-size:0.55rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#94a3b8">♻️ Score</div>
  <div id="scoreVal" style="font-size:1.8rem;font-weight:800;font-variant-numeric:tabular-nums;background:linear-gradient(135deg,#22c55e,#86efac);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;transition:transform 0.15s">0</div>
</div>

<!-- Header -->
<div class="header">
  <h1>🤖 GCBOT Control</h1>
  <div class="sub"><span class="status-dot"></span>Live stream &amp; servo control</div>
</div>

<!-- Landscape Layout -->
<div class="landscape-container">
  
  <!-- Left: Movement Controls -->
  <div class="landscape-left">
    <div class="panel">
      <div class="panel-title">Movement</div>

      <div class="dpad">
        <button class="ctrl-btn move fwd"   id="btnFwd"   title="Forward (W)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg></button>
        <button class="ctrl-btn move left"  id="btnLeft"  title="Left (A)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5"/><path d="m12 19-7-7 7-7"/></svg></button>
        <div></div>
        <button class="ctrl-btn move right" id="btnRight" title="Right (D)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg></button>
        <button class="ctrl-btn move back"  id="btnBack"  title="Back (S)"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg></button>
      </div>

      <div style="font-size:0.6rem;color:var(--muted);margin-top:2px">WASD</div>

      <!-- DROP button -->
      <button class="drop-obj-btn" id="btnDropObj" title="Release gripper to 50° then check weight">
        🗑️ DROP OBJECT
      </button>
    </div>
  </div>

  <!-- Center: Video Stream -->
  <div class="landscape-center">
    <div class="stream-card" style="position:relative">
      <img id="stream" src="/video_feed" alt="Live Stream">
      <!-- Crosshair overlay -->
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-60%);pointer-events:none;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0">
        <div style="width:2px;height:16px;background:rgba(255,255,255,0.75);border-radius:1px"></div>
        <div style="display:flex;align-items:center">
          <div style="width:16px;height:2px;background:rgba(255,255,255,0.75);border-radius:1px"></div>
          <div style="width:6px;height:6px;border-radius:50%;border:2px solid rgba(255,255,255,0.9);margin:0 2px"></div>
          <div style="width:16px;height:2px;background:rgba(255,255,255,0.75);border-radius:1px"></div>
        </div>
        <div style="width:2px;height:16px;background:rgba(255,255,255,0.75);border-radius:1px"></div>
      </div>
      <div class="stream-label">
        <span style="display:flex;align-items:center;gap:6px">
          <span class="status-dot" style="width:5px;height:5px"></span>
          <span id="feedName">Live Camera Feed</span>
        </span>
        <div class="feed-toggle">
          <button id="btnRaw"      onclick="setFeed('raw')">Raw</button>
          <button id="btnDetected" onclick="setFeed('detected')">AI Detection</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Right: Gripper and Arm Lift -->
  <div class="landscape-right">
    
    <!-- Gripper Panel -->
    <div class="panel">
      <div class="panel-title">Gripper</div>

      <div class="ring-wrap" id="gripRing">
        <svg viewBox="0 0 100 100">
          <circle class="ring-bg" cx="50" cy="50" r="44"/>
          <circle class="ring-fg" cx="50" cy="50" r="44"
                  stroke="var(--green)"
                  stroke-dasharray="276.46"
                  stroke-dashoffset="276.46"
                  id="gripArc"/>
        </svg>
        <div class="ring-value">
          <span class="angle-display" id="gripAngle">90</span>
          <span class="angle-unit">deg</span>
        </div>
      </div>

      <div class="btn-row">
        <button class="ctrl-btn grab"    id="btnGrab"    title="Grab (hold)"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 11V6a2 2 0 0 0-4 0v2"/><path d="M14 10V4a2 2 0 0 0-4 0v6"/><path d="M10 10.5V6a2 2 0 0 0-4 0v8"/><path d="M18 11a2 2 0 1 1 4 0v3a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg></button>
        <button class="ctrl-btn release" id="btnRelease" title="Release"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 11V6a2 2 0 0 0-4 0v5"/><path d="M14 10V4a2 2 0 0 0-4 0v6"/><path d="M10 10V6a2 2 0 0 0-4 0v8"/><path d="M18 11a2 2 0 1 1 4 0v3a8 8 0 0 1-8 8h-4a8 8 0 0 1-8-8V9"/></svg></button>
      </div>
    </div>

    <!-- Arm Lift Panel -->
    <div class="panel">
      <div class="panel-title">Arm Lift</div>

      <div class="ring-wrap" id="liftRing">
        <svg viewBox="0 0 100 100">
          <circle class="ring-bg" cx="50" cy="50" r="44"/>
          <circle class="ring-fg" cx="50" cy="50" r="44"
                  stroke="var(--blue)"
                  stroke-dasharray="276.46"
                  stroke-dashoffset="276.46"
                  id="liftArc"/>
        </svg>
        <div class="ring-value">
          <span class="angle-display" id="liftAngle">90</span>
          <span class="angle-unit">deg</span>
        </div>
      </div>

      <div class="btn-row">
        <button class="ctrl-btn lift" id="btnLift" title="Lift (hold)"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg></button>
        <button class="ctrl-btn drop" id="btnDrop" title="Drop (hold)"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg></button>
      </div>
    </div>

  </div>

</div>

<div class="footer">Hold buttons for smooth movement · WASD for driving · Release grip snaps 50° → 90°</div>

<script>
/* ─── Feed selector ─── */
function setFeed(mode) {
  const img  = document.getElementById('stream');
  const name = document.getElementById('feedName');
  const btnR = document.getElementById('btnRaw');
  const btnD = document.getElementById('btnDetected');
  if (mode === 'detected') {
    img.src  = '/video_feed_detected';
    name.textContent = 'AI Detection Feed';
    btnD.classList.add('active');
    btnR.classList.remove('active');
  } else {
    img.src  = '/video_feed';
    name.textContent = 'Live Camera Feed';
    btnR.classList.add('active');
    btnD.classList.remove('active');
  }
}

// Default: detected feed on host, raw on remote
(function() {
  const h = window.location.hostname;
  setFeed((h === 'localhost' || h === '127.0.0.1') ? 'detected' : 'raw');
})();

/* ─── helpers ─── */
const CIRCUMFERENCE = 2 * Math.PI * 44;  // ≈ 276.46

function sendCmd(cmd) {
  fetch('/cmd', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({cmd})
  }).catch(()=>{});
}

function updateRing(arcEl, angle, max) {
  const pct = angle / max;
  arcEl.style.strokeDashoffset = CIRCUMFERENCE * (1 - pct);
}

/* ─── state ─── */
let gripAngle = 90;
let liftAngle = 90;
const GRIP_MAX  = 175;
const GRIP_MIN  = 0;
const LIFT_MAX  = 180;
const LIFT_MIN  = 0;
const INTERVAL       = 20;   // ms per step for lift buttons
const GRIP_INTERVAL  = 80;   // ms per step for gripper — slower, more precise

const gripDisplay = document.getElementById('gripAngle');
const liftDisplay = document.getElementById('liftAngle');
const gripArc     = document.getElementById('gripArc');
const liftArc     = document.getElementById('liftArc');

function refreshGripUI() {
  gripDisplay.textContent = gripAngle;
  updateRing(gripArc, gripAngle, GRIP_MAX);
}
function refreshLiftUI() {
  liftDisplay.textContent = liftAngle;
  updateRing(liftArc, liftAngle, LIFT_MAX);
}
refreshGripUI();
refreshLiftUI();

/* ─── hold-to-move engine ─── */
function holdAction(btnEl, tick, interval) {
  interval = interval || INTERVAL;
  let iv = null;
  const start = (e) => {
    e.preventDefault();
    btnEl.classList.add('active');
    tick();                       // immediate first tick
    iv = setInterval(tick, interval);
  };
  const stop = () => {
    btnEl.classList.remove('active');
    if (iv) { clearInterval(iv); iv = null; }
  };
  btnEl.addEventListener('mousedown',   start);
  btnEl.addEventListener('touchstart',  start, {passive:false});
  btnEl.addEventListener('mouseup',     stop);
  btnEl.addEventListener('mouseleave',  stop);
  btnEl.addEventListener('touchend',    stop);
  btnEl.addEventListener('touchcancel', stop);
}

/* ─── DROP OBJECT button ─── */
document.getElementById('btnDropObj').addEventListener('click', () => {
  const btn = document.getElementById('btnDropObj');
  if (btn.classList.contains('busy')) return;

  // 1. Open gripper to 50°
  gripAngle = 50;
  sendCmd('G50');
  refreshGripUI();

  // 2. Disable button & show countdown
  btn.classList.add('busy');
  btn.textContent = '⏳ Waiting 2s…';

  // 3. After 2s, send weight-check trigger
  setTimeout(() => {
    sendCmd('WEIGHTNOW');
    btn.textContent = '⚖️ Weighing…';

    // 4. Poll for score update (up to 5s)
    const startScore = _lastScore;
    let checks = 0;
    const checker = setInterval(() => {
      checks++;
      fetch('/score').then(r => r.json()).then(d => {
        if (d.score !== startScore) {
          clearInterval(checker);
          _lastScore = d.score;
          const ev = d.event;
          const el = document.getElementById('scoreVal');
          if (el) { el.textContent = d.score; el.classList.add('bump'); setTimeout(() => el.classList.remove('bump'), 300); }
          btn.textContent = `+${ev.pts} ${ev.cls} (${ev.weight}g)`;
          setTimeout(() => { btn.classList.remove('busy'); btn.innerHTML = '🗑️ DROP OBJECT'; }, 2500);
        } else if (checks >= 10) {
          clearInterval(checker);
          btn.textContent = 'No score (too light?)';
          setTimeout(() => { btn.classList.remove('busy'); btn.innerHTML = '🗑️ DROP OBJECT'; }, 1500);
        }
      }).catch(() => {});
    }, 500);
  }, 2000);
});


holdAction(document.getElementById('btnGrab'), () => {
  if (gripAngle < GRIP_MAX) {
    gripAngle += 1;
    sendCmd('G' + gripAngle);
    refreshGripUI();
  }
}, GRIP_INTERVAL);

/* ─── Release (hold → decrement gripper by 1 per tick) ─── */
holdAction(document.getElementById('btnRelease'), () => {
  if (gripAngle > GRIP_MIN) {
    gripAngle -= 1;
    sendCmd('G' + gripAngle);
    refreshGripUI();
  }
}, GRIP_INTERVAL);

/* ─── Lift (↑ button → DECREASE angle) ─── */
holdAction(document.getElementById('btnLift'), () => {
  if (liftAngle > LIFT_MIN) {
    liftAngle -= 1;
    sendCmd('U' + liftAngle);
    refreshLiftUI();
  }
});

/* ─── Drop (↓ button → INCREASE angle) ─── */
holdAction(document.getElementById('btnDrop'), () => {
  if (liftAngle < LIFT_MAX) {
    liftAngle += 1;
    sendCmd('U' + liftAngle);
    refreshLiftUI();
  }
});

/* ─── Movement — Pointer Events only (no double-fire on mobile) ─── */
// Pointer Events API unifies mouse + touch in a single event stream.
// Using ONLY pointerdown/pointerup avoids the double-fire issue where
// both touchstart AND pointerdown fire on mobile.

let _activePointerId = null;   // which pointer is holding a move button
let _activeMovBtn    = null;

// Throttle: skip sendCmd if same cmd sent within THROTTLE_MS
const THROTTLE_MS = 80;
let _lastCmdTime = 0;
let _lastCmdVal  = null;
function sendCmdThrottled(cmd) {
  const now = Date.now();
  if (cmd === _lastCmdVal && now - _lastCmdTime < THROTTLE_MS) return;
  _lastCmdTime = now;
  _lastCmdVal  = cmd;
  sendCmd(cmd);
}

function _stopMove() {
  if (_activeMovBtn) {
    _activeMovBtn.classList.remove('active');
    _activeMovBtn    = null;
    _activePointerId = null;
    sendCmdThrottled('S');
  }
}

// Safety net: stop robot if user switches tab / minimises app
document.addEventListener('visibilitychange', () => { if (document.hidden) _stopMove(); });
window.addEventListener('blur', _stopMove);

function moveHold(btnEl, cmdDown) {
  btnEl.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    if (_activePointerId !== null) return;  // another pointer already active
    _activePointerId = e.pointerId;
    _activeMovBtn    = btnEl;
    btnEl.classList.add('active');
    btnEl.setPointerCapture(e.pointerId);   // keep receiving events even if finger drifts off
    sendCmdThrottled(cmdDown);
  });

  btnEl.addEventListener('pointerup',     _stopMove);
  btnEl.addEventListener('pointercancel', _stopMove);
  // Stop if finger slides off button
  btnEl.addEventListener('pointerleave',  (e) => {
    if (e.pointerId === _activePointerId) _stopMove();
  });
}

moveHold(document.getElementById('btnFwd'),   'B');
moveHold(document.getElementById('btnBack'),  'F');
moveHold(document.getElementById('btnLeft'),  'L');
moveHold(document.getElementById('btnRight'), 'R');

/* ─── Keyboard controls (WASD) ─── */
const keyMap   = {w:'F', W:'F', a:'L', A:'L', s:'B', S:'B', d:'R', D:'R'};
const btnIdMap = {w:'btnFwd', a:'btnLeft', s:'btnBack', d:'btnRight'};
const keysDown = new Set();

window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (keyMap[e.key] && !keysDown.has(k)) {
    keysDown.add(k);
    sendCmdThrottled(keyMap[e.key]);
    const id = btnIdMap[k];
    if (id) document.getElementById(id).classList.add('active');
  }
});
window.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  if (keyMap[e.key] && keysDown.has(k)) {
    keysDown.delete(k);
    // Only send stop if no other direction key is still held
    if (keysDown.size === 0) sendCmdThrottled('S');
    else sendCmdThrottled(keyMap[Array.from(keysDown).find(kk => keyMap[kk])] || 'S');
    const id = btnIdMap[k];
    if (id) document.getElementById(id).classList.remove('active');
  }
});

/* ─── Score polling ─── */
let _lastScore = 0;
function pollScore() {
  fetch('/score').then(r => r.json()).then(d => {
    const el = document.getElementById('scoreVal');
    if (el && d.score !== _lastScore) {
      _lastScore = d.score;
      el.textContent = d.score;
      el.classList.add('bump');
      setTimeout(() => el.classList.remove('bump'), 300);
    }
  }).catch(()=>{});
}
setInterval(pollScore, 1500);
pollScore();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n✦  GCBOT Control running → http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
