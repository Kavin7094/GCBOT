# -*- coding: utf-8 -*-
import socket
import serial
import threading
import time

# -------- Serial settings --------
ARDUINO_PORT = "/dev/ttyUSB0"   # change if needed
ARDUINO_BAUD = 9600             # increase to 115200 on both Pi & Arduino for lower latency

# -------- TCP server settings --------
HOST = "0.0.0.0"
PORT = 5000

# -------- Shared state --------
arduino = None
clients = []
clients_lock = threading.Lock()

# Score is ONLY updated when the Arduino sends SCORE:N in response
# to an explicit WEIGHTNOW command.  The Pi never calculates weight
# or score autonomously.
latest_score = 0


def broadcast(message: str):
    """Send a message to all connected laptop clients."""
    dead_clients = []

    with clients_lock:
        for conn in clients:
            try:
                conn.sendall((message + "\n").encode())
            except Exception:
                dead_clients.append(conn)

        for conn in dead_clients:
            try:
                clients.remove(conn)
            except ValueError:
                pass


def send_to_arduino(cmd: str):
    """Send a newline-terminated command to Arduino."""
    global arduino

    if arduino is None:
        print("Arduino not connected, cannot send:", cmd)
        return

    try:
        payload = (cmd + "\n").encode()
        arduino.write(payload)
        arduino.flush()
    except Exception as e:
        print("Failed to send to Arduino:", e)


def serial_reader():
    """
    Continuously read Arduino output.

    Weight/score is calculated ONLY on the Arduino side, triggered by the
    WEIGHTNOW command.  The Pi simply forwards SCORE:N messages it receives.
    W: debug lines are printed but never used for scoring here.
    """
    global arduino, latest_score

    while True:
        try:
            if arduino is None:
                time.sleep(1)
                continue

            # Blocking readline — reacts immediately, no 50ms polling delay
            line = arduino.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            print("Arduino:", line)

            # Arduino scored - forward to all laptop clients immediately
            if line.startswith("SCORE:"):
                try:
                    latest_score = int(line.split(":", 1)[1])
                except ValueError:
                    pass
                broadcast(line)   # e.g. "SCORE:10"

            # Weight reading — forward to laptop so it can compute combined score
            elif line.startswith("W:"):
                broadcast(line)   # e.g. "W:23.45"

            # else: other debug lines from Arduino — logged above, ignored here

        except Exception as e:
            print("Serial read error:", e)
            time.sleep(1)


def handle_client(conn, addr):
    """
    Laptop client protocol:
    - send motor/servo commands like F, B, L, R, S, U90, G120
    - receive score updates like SCORE:5
    """
    print("Control client connected:", addr)

    with clients_lock:
        clients.append(conn)

    try:
        conn.sendall((f"SCORE:{latest_score}\n").encode())

        buffer = b""
        while True:
            data = conn.recv(1024)
            if not data:
                print("Control client disconnected:", addr)
                break

            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                cmd = line.decode().strip()

                if cmd:
                    print("Received command from laptop:", cmd)
                    send_to_arduino(cmd)

    except Exception as e:
        print("Control client error:", e)

    finally:
        with clients_lock:
            if conn in clients:
                clients.remove(conn)
        try:
            conn.close()
        except Exception:
            pass


def start_serial():
    """Open Arduino serial."""
    global arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)
        print("Arduino serial opened on", ARDUINO_PORT)
    except Exception as e:
        print("ERROR opening serial:", e)
        arduino = None


def start_server():
    """TCP server for laptop control."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)

    print(f"Command server listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        # FIX: Disable Nagle's algorithm — sends small commands immediately
        # without this, the OS may buffer "F\n" for up to ~200ms before transmitting
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        t.start()


if __name__ == "__main__":
    start_serial()
    threading.Thread(target=serial_reader, daemon=True).start()
    start_server()
