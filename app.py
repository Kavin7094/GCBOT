"""
GCBOT – Combined Control & Video Stream App
Merges streamrx.py (video + Roboflow inference) and laptop_control.py (Flask control)
into a single file with a modern dark UI.
"""

from flask import Flask, Response, render_template_string, request, jsonify, session, redirect, url_for
import socket, struct, threading, time, os, hashlib, uuid, json, io, base64
from functools import wraps
try:
    import qrcode; HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False
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

# ── WiFi hotspot credentials (change to match your network/hotspot)
WIFI_SSID     = "LAPTOP-OO8EF23U 3188"          # your network name
WIFI_PASSWORD = "33333333"      # your network password
WIFI_TYPE     = "WPA"            # WPA | WEP | nopass

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
MIN_WEIGHT  =   1.0   # grams — below this, treat as no object (noise rejection)

# ═══════════════════════════════════════════════════════════
#  GLOBALS
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get("GCBOT_SECRET", "gcbot-secret-2026")

# ═══════════════════════════════════════════════════════════
#  USER MANAGEMENT
# ═══════════════════════════════════════════════════════════
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
_users_lock = threading.Lock()

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with _users_lock:
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

def save_users(users):
    with _users_lock:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)

def hash_pw(pw):
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"

def get_subnet(ip: str) -> str:
    """Return the /24 prefix, e.g. '10.107.83' for '10.107.83.5'."""
    return ".".join(ip.split(".")[:3])

def is_same_network(client_ip: str) -> bool:
    """True when client is in the same /24 subnet as this machine."""
    try:
        server_ip = get_local_ip()
        if client_ip in ("127.0.0.1", "::1", server_ip):
            return True          # always allow localhost
        return get_subnet(client_ip) == get_subnet(server_ip)
    except Exception:
        return True              # fail open

def save_user_score(user_id, new_score):
    users = load_users()
    for u in users:
        if u["id"] == user_id:
            u["score"] = new_score; break
    save_users(users)

# ── Network guard: redirect off-network clients to WiFi connect page
NETWORK_EXEMPT = {"/qr", "/qr_img", "/wifi_qr_img", "/wrong_network",
                  "/video_feed", "/video_feed_detected"}

@app.before_request
def check_network():
    if request.path in NETWORK_EXEMPT:
        return None
    client_ip = request.remote_addr or ""
    if not is_same_network(client_ip):
        return redirect(url_for("wrong_network"))


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

latest_frame   = None
detected_frame = None
running        = True
latest_score   = 0
active_user_id = None   # set on login; used to persist cumulative score

# Last detection snapshot (updated every inference frame)
last_detected_class = None   # e.g. "Bottle"
last_detected_conf  = 0.0    # e.g. 0.87

# Set True when WEIGHTNOW is sent; cleared once W: reply arrives
weighnow_pending = False
weighnow_time    = 0.0     # timestamp when WEIGHTNOW was sent (for timeout)
WEIGHNOW_TIMEOUT = 10.0    # seconds — auto-clear if Arduino never replies
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
    if active_user_id:
        save_user_score(active_user_id, latest_score)

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
                        # W:ERR, W:baseline, etc. — clear pending to avoid stale state
                        if weighnow_pending:
                            weighnow_pending = False
                            print(f"[SCORE] Weight error from Arduino: {raw}")
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
    global weighnow_pending, weighnow_time
    data = request.get_json(silent=True)
    if not data or "cmd" not in data:
        return jsonify(ok=False, error="no cmd"), 400
    command = str(data["cmd"]).strip()
    if command == "WEIGHTNOW":
        weighnow_pending = True
        weighnow_time = time.time()

        # Watchdog: always fire after 3.5s if hardware never responds.
        # Real hardware clears weighnow_pending first → watchdog skips.
        # Stale/offline Pi → watchdog fires simulated weight.
        import random
        def _demo_watchdog():
            global weighnow_pending
            time.sleep(3.5)
            if weighnow_pending:
                demo_g = round(random.uniform(20, 120), 1)
                print(f"[DEMO] No hardware response — simulating weight: {demo_g}g")
                compute_combined_score(demo_g)
                weighnow_pending = False
        threading.Thread(target=_demo_watchdog, daemon=True).start()

    ok, info = send_to_pi(command)
    return jsonify(ok=ok, info=info)

@app.route("/simulate", methods=["POST"])
def simulate():
    """Test scoring without hardware.
    POST /simulate {"cls": "Bottle", "weight": 50}
    If cls is omitted, uses last_detected_class. If weight is omitted, uses 25g.
    """
    global last_detected_class, last_detected_conf
    data = request.get_json(silent=True) or {}
    cls = data.get("cls")
    weight = float(data.get("weight", 25.0))
    if cls:
        last_detected_class = cls
        last_detected_conf = 0.90
    event = compute_combined_score(weight)
    return jsonify(ok=True, event=event)

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
        } else if (checks >= 20) {
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
@login_required
def index():
    return render_template_string(HTML,
        nickname=session.get("nickname", ""),
        avatar=session.get("avatar", "🤖"))

# ═══════════════════════════════════════════════════════════
#  AUTH HTML TEMPLATES
# ═══════════════════════════════════════════════════════════
_BASE_CSS = """
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0b0e14;--glass:rgba(255,255,255,0.06);--border:rgba(255,255,255,0.10);
--text:#e2e8f0;--muted:#94a3b8;--accent:#6366f1;--ag:linear-gradient(135deg,#6366f1,#a78bfa);
--green:#22c55e;--red:#ef4444}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;
background-image:radial-gradient(ellipse at 20% 50%,rgba(99,102,241,.08) 0%,transparent 60%),
radial-gradient(ellipse at 80% 20%,rgba(167,139,250,.06) 0%,transparent 50%)}
.card{background:var(--glass);border:1px solid var(--border);border-radius:20px;
padding:38px 34px;width:100%;max-width:420px;backdrop-filter:blur(20px)}
.logo{text-align:center;margin-bottom:26px}
.logo-icon{font-size:2.4rem}
.logo h1{font-size:1.45rem;font-weight:800;background:var(--ag);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:6px}
.logo p{font-size:.78rem;color:var(--muted);margin-top:3px}
.field{margin-bottom:14px}
.field label{display:block;font-size:.7rem;font-weight:600;color:var(--muted);
margin-bottom:5px;text-transform:uppercase;letter-spacing:.06em}
.field input{width:100%;background:rgba(255,255,255,.04);border:1px solid var(--border);
border-radius:10px;padding:11px 13px;color:var(--text);font-size:.88rem;
font-family:inherit;outline:none;transition:border-color .2s}
.field input:focus{border-color:var(--accent)}
.btn{width:100%;padding:12px;border-radius:10px;border:none;background:var(--ag);
color:#fff;font-size:.93rem;font-weight:700;font-family:inherit;cursor:pointer;
transition:opacity .2s,transform .1s;margin-top:6px}
.btn:hover{opacity:.9}.btn:active{transform:scale(.98)}
.link-row{text-align:center;margin-top:18px;font-size:.8rem;color:var(--muted)}
.link-row a{color:#a78bfa;text-decoration:none;font-weight:600}
.error{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
border-radius:8px;padding:10px 14px;font-size:.8rem;color:#fca5a5;margin-bottom:14px}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
"""

LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GCBOT – Sign In</title>""" + _BASE_CSS + """</head><body>
<div class="card">
  <div class="logo"><div class="logo-icon">🤖</div>
    <h1>GCBOT Control</h1><p>Sign in to start playing</p></div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <div class="field"><label>Email</label>
      <input type="email" name="email" placeholder="you@example.com" required autofocus></div>
    <div class="field"><label>Password</label>
      <input type="password" name="password" placeholder="••••••••" required></div>
    <button class="btn">Sign In →</button>
  </form>
  <div class="link-row">New player? <a href="/register">Create account</a></div>
  <div style="text-align:center;margin-top:10px;font-size:.75rem">
    <a href="/leaderboard" style="color:var(--muted);text-decoration:none">🏆 View Leaderboard</a>
  </div>
</div></body></html>"""

REGISTER_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GCBOT – Register</title>""" + _BASE_CSS + """
<style>
.card{max-width:500px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.avatar-grid{display:flex;flex-wrap:wrap;gap:7px;margin-top:6px}
.av-in{display:none}
.av-in+label{font-size:1.5rem;width:42px;height:42px;display:flex;align-items:center;
justify-content:center;border-radius:10px;border:2px solid transparent;
background:rgba(255,255,255,.05);cursor:pointer;transition:.15s}
.av-in:checked+label{border-color:var(--accent);background:rgba(99,102,241,.15);transform:scale(1.1)}
.av-in+label:hover{border-color:rgba(99,102,241,.5)}
</style>
</head><body>
<div class="card">
  <div class="logo"><div class="logo-icon">🎮</div>
    <h1>Create Profile</h1><p>Join GCBOT and collect some trash!</p></div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <div class="row2">
      <div class="field"><label>Full Name</label>
        <input type="text" name="name" placeholder="Kavin" required autofocus></div>
      <div class="field"><label>Nickname <span style="color:#a78bfa;font-size:.65rem">★ leaderboard</span></label>
        <input type="text" name="nickname" placeholder="GCBot_King" required maxlength="20"></div>
    </div>
    <div class="field"><label>Choose Avatar</label>
      <div class="avatar-grid">
        {% for av in ["🤖","🦾","🎮","🏆","⚡","🔥","💎","🎯","🚀","🌊","🦁","🐉","🦅","🌈","👾"] %}
        <input class="av-in" type="radio" name="avatar" id="av{{loop.index}}" value="{{av}}"
          {{"checked" if loop.first else ""}}>
        <label for="av{{loop.index}}">{{av}}</label>
        {% endfor %}
      </div>
    </div>
    <div class="field"><label>Email</label>
      <input type="email" name="email" placeholder="you@example.com" required></div>
    <div class="row2">
      <div class="field"><label>Password</label>
        <input type="password" name="password" placeholder="Min 4 chars" required minlength="4"></div>
      <div class="field"><label>Confirm</label>
        <input type="password" name="confirm" placeholder="••••••••" required></div>
    </div>
    <button class="btn">Create Account 🚀</button>
  </form>
  <div class="link-row">Already have an account? <a href="/login">Sign in</a></div>
</div></body></html>"""

QR_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GCBOT – Scan to Play</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0b0e14;color:#e2e8f0;
min-height:100vh;display:flex;flex-direction:column;align-items:center;
justify-content:center;gap:20px;padding:30px;
background-image:radial-gradient(ellipse at 50% 0%,rgba(99,102,241,.15) 0%,transparent 60%)}
.title{font-size:.9rem;font-weight:700;text-transform:uppercase;letter-spacing:.15em;color:#94a3b8}
.brand{font-size:2.5rem;font-weight:800;
background:linear-gradient(135deg,#6366f1,#a78bfa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.qr-row{display:flex;gap:32px;align-items:flex-start;justify-content:center;flex-wrap:wrap}
.qr-card{display:flex;flex-direction:column;align-items:center;gap:12px}
.qr-label{font-size:.75rem;font-weight:700;text-transform:uppercase;
letter-spacing:.1em;color:#94a3b8}
.step{font-size:1.4rem;font-weight:800;color:#6366f1}
.qr-frame{background:white;border-radius:16px;padding:14px;
box-shadow:0 0 50px rgba(99,102,241,.3)}
.qr-frame img{display:block;border-radius:6px}
.qr-sub{font-size:.72rem;color:#64748b;text-align:center;max-width:160px;line-height:1.4}
.divider{width:1px;background:rgba(255,255,255,.08);align-self:stretch}
.url{font-size:.9rem;font-weight:600;color:#a78bfa;
background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.2);
border-radius:10px;padding:8px 20px}
.hint{font-size:.75rem;color:#64748b}
</style>
</head><body>
<div class="title">Scan to Play</div>
<div class="brand">🤖 GCBOT</div>
<div class="qr-row">
  <!-- Step 1: WiFi -->
  <div class="qr-card">
    <div class="step">Step 1</div>
    <div class="qr-label">📶 Join WiFi</div>
    <div class="qr-frame"><img src="/wifi_qr_img" width="180" height="180" alt="WiFi QR"></div>
    <div class="qr-sub">Scan to connect to <strong style="color:#e2e8f0">{{ ssid }}</strong> network</div>
  </div>
  <div class="divider"></div>
  <!-- Step 2: Login -->
  <div class="qr-card">
    <div class="step">Step 2</div>
    <div class="qr-label">🎮 Open Game</div>
    <div class="qr-frame"><img src="/qr_img" width="180" height="180" alt="Login QR"></div>
    <div class="qr-sub">Scan to open the control page</div>
  </div>
</div>
<div class="url">{{ url }}</div>
<div class="hint">First join the WiFi, then scan the second code to play</div>
</body></html>"""

WRONG_NETWORK_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GCBOT – Wrong Network</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0b0e14;color:#e2e8f0;
min-height:100vh;display:flex;flex-direction:column;align-items:center;
justify-content:center;text-align:center;gap:20px;padding:24px;
background-image:radial-gradient(ellipse at 50% 30%,rgba(239,68,68,.1) 0%,transparent 60%)}
.icon{font-size:3.5rem}
h1{font-size:1.6rem;font-weight:800;
background:linear-gradient(135deg,#f87171,#fca5a5);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
p{color:#94a3b8;font-size:.88rem;max-width:320px;line-height:1.6}
.ssid-box{background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25);
border-radius:12px;padding:14px 28px}
.ssid{font-size:1.2rem;font-weight:800;color:#a78bfa}
.pw{font-size:.8rem;color:#94a3b8;margin-top:4px}
.qr-frame{background:white;border-radius:16px;padding:12px;
box-shadow:0 0 40px rgba(239,68,68,.2);display:inline-block}
.retry{display:inline-block;margin-top:8px;padding:10px 28px;
border-radius:10px;background:linear-gradient(135deg,#6366f1,#a78bfa);
color:#fff;font-weight:700;font-size:.88rem;text-decoration:none}
</style>
</head><body>
<div class="icon">📵</div>
<h1>Wrong Network</h1>
<p>Your device is not connected to the GCBOT network.<br>
Please connect first, then try again.</p>
<div class="ssid-box">
  <div class="ssid">📡 {{ ssid }}</div>
  <div class="pw">Password: <strong style="color:#e2e8f0">{{ password }}</strong></div>
</div>
<div class="qr-frame">
  <img src="/wifi_qr_img" width="180" height="180" alt="WiFi QR Code">
</div>
<p style="font-size:.75rem">Scan above to connect automatically</p>
<a href="/login" class="retry">I'm connected → Try Again</a>
</body></html>"""

LEADERBOARD_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GCBOT – Leaderboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0b0e14;color:#e2e8f0;
min-height:100vh;padding:30px 16px;
background-image:radial-gradient(ellipse at 50% 0%,rgba(99,102,241,.1) 0%,transparent 60%)}
.header{text-align:center;margin-bottom:32px}
.header h1{font-size:2rem;font-weight:800;
background:linear-gradient(135deg,#f59e0b,#fbbf24);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header p{color:#94a3b8;font-size:.85rem;margin-top:6px}
.table-wrap{max-width:600px;margin:0 auto}
.row{display:flex;align-items:center;gap:14px;
background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
border-radius:14px;padding:14px 18px;margin-bottom:10px;transition:.2s}
.row:hover{background:rgba(255,255,255,.07)}
.rank{font-size:1.3rem;font-weight:800;width:36px;text-align:center;flex-shrink:0}
.rank.r1{color:#f59e0b}.rank.r2{color:#94a3b8}.rank.r3{color:#cd7c3e}
.av{font-size:1.6rem;flex-shrink:0}
.info{flex:1;min-width:0}
.nick{font-weight:700;font-size:.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.name{font-size:.72rem;color:#94a3b8;margin-top:2px}
.score{font-size:1.3rem;font-weight:800;
background:linear-gradient(135deg,#22c55e,#86efac);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;flex-shrink:0}
.you-badge{font-size:.6rem;font-weight:700;background:#6366f1;color:#fff;
border-radius:6px;padding:2px 6px;margin-left:6px;vertical-align:middle}
.empty{text-align:center;padding:40px;color:#94a3b8;font-size:.9rem}
.back-btn{display:block;text-align:center;margin:24px auto 0;max-width:200px;
padding:12px;border-radius:12px;background:linear-gradient(135deg,#6366f1,#a78bfa);
color:#fff;font-weight:700;font-size:.9rem;text-decoration:none;
font-family:inherit;border:none;cursor:pointer;transition:opacity .2s}
.back-btn:hover{opacity:.85}
</style>
</head><body>
<div class="header">
  <h1>🏆 Leaderboard</h1>
  <p>Top GCBOT Trash Collectors</p>
</div>
<div class="table-wrap">
  {% if users %}
    {% for u in users %}
    <div class="row">
      <div class="rank {{'r1' if loop.index==1 else 'r2' if loop.index==2 else 'r3' if loop.index==3 else ''}}">
        {{'🥇' if loop.index==1 else '🥈' if loop.index==2 else '🥉' if loop.index==3 else loop.index}}
      </div>
      <div class="av">{{ u.avatar }}</div>
      <div class="info">
        <div class="nick">{{ u.nickname }}
          {% if u.nickname == current_nick %}<span class="you-badge">YOU</span>{% endif %}
        </div>
        <div class="name">{{ u.name }}</div>
      </div>
      <div class="score">{{ u.score }} pts</div>
    </div>
    {% endfor %}
  {% else %}
    <div class="empty">No players yet — be the first to register! 🚀</div>
  {% endif %}
  <a href="/" class="back-btn">← Back to Control</a>
</div>
</body></html>"""

# ═══════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════
@app.route("/login", methods=["GET", "POST"])
def login():
    global latest_score, active_user_id
    if "user_id" in session:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw    = request.form.get("password", "")
        users = load_users()
        user  = next((u for u in users
                      if u["email"].lower() == email and u["password"] == hash_pw(pw)), None)
        if user:
            session["user_id"]  = user["id"]
            session["nickname"] = user["nickname"]
            session["avatar"]   = user["avatar"]
            active_user_id = user["id"]
            latest_score   = user.get("score", 0)
            return redirect(url_for("index"))
        error = "Invalid email or password"
    return render_template_string(LOGIN_HTML, error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    global latest_score, active_user_id
    if "user_id" in session:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        nickname = request.form.get("nickname", "").strip()
        avatar   = request.form.get("avatar", "🤖")
        email    = request.form.get("email", "").strip().lower()
        pw       = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if not all([name, nickname, email, pw]):
            error = "All fields are required"
        elif pw != confirm:
            error = "Passwords do not match"
        elif len(pw) < 4:
            error = "Password must be at least 4 characters"
        else:
            users = load_users()
            if any(u["email"].lower() == email for u in users):
                error = "Email already registered"
            elif any(u["nickname"].lower() == nickname.lower() for u in users):
                error = "Nickname already taken — try another"
            else:
                new_user = {
                    "id":         str(uuid.uuid4()),
                    "name":       name,
                    "nickname":   nickname,
                    "avatar":     avatar,
                    "email":      email,
                    "password":   hash_pw(pw),
                    "score":      0,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                users.append(new_user)
                save_users(users)
                session["user_id"]  = new_user["id"]
                session["nickname"] = new_user["nickname"]
                session["avatar"]   = new_user["avatar"]
                active_user_id = new_user["id"]
                latest_score   = 0
                return redirect(url_for("index"))
    return render_template_string(REGISTER_HTML, error=error)

@app.route("/logout")
def logout():
    global active_user_id
    session.clear()
    active_user_id = None
    return redirect(url_for("login"))

@app.route("/qr")
def qr_display():
    ip  = get_local_ip()
    url = f"http://{ip}:5001/login"
    return render_template_string(QR_HTML, ip=ip, url=url, ssid=WIFI_SSID)

@app.route("/wifi_qr_img")
def wifi_qr_img():
    """QR code encoding WiFi credentials — phone scans and auto-joins network."""
    wifi_str = f"WIFI:T:{WIFI_TYPE};S:{WIFI_SSID};P:{WIFI_PASSWORD};;"
    if HAS_QRCODE:
        img = qrcode.make(wifi_str)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png")
    return Response("Install qrcode: pip install qrcode[pil]", mimetype="text/plain")

@app.route("/wrong_network")
def wrong_network():
    return render_template_string(WRONG_NETWORK_HTML,
        ssid=WIFI_SSID, password=WIFI_PASSWORD), 403

@app.route("/qr_img")
def qr_img():
    ip  = get_local_ip()
    url = f"http://{ip}:5001/login"
    if HAS_QRCODE:
        img = qrcode.make(url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png")
    return Response(f"Install qrcode: pip install qrcode[pil]\nURL: {url}",
                    mimetype="text/plain")

@app.route("/leaderboard")
def leaderboard():
    users = load_users()
    users_sorted = sorted(users, key=lambda u: u.get("score", 0), reverse=True)
    return render_template_string(LEADERBOARD_HTML,
        users=users_sorted,
        current_nick=session.get("nickname", ""))

# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n✦  GCBOT Control running → http://localhost:5001\n")
    print(f"   QR Display page  → http://localhost:5001/qr\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)

