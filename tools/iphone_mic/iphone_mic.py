#!/usr/bin/env python3
"""iPhone -> Linux microphone bridge (stdlib only).

Open https://<this-host>:8443/ in Safari on the iPhone, tap Start, and the
phone's mic appears in PipeWire as the source "iPhone Mic".

Pipeline: Safari getUserMedia -> Int16 PCM over WebSocket -> pw-cat ->
virtual sink "iphone_mic_sink" -> pw-loopback -> virtual source "iPhone Mic".
"""
import argparse, base64, hashlib, http.server, os, shutil, signal, socket, ssl
import struct, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONF = Path.home() / ".config" / "iphone-mic"
SINK = "iphone_mic_sink"
SOURCE = "iphone_mic"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ensure_cert(hosts):
    CONF.mkdir(parents=True, exist_ok=True)
    crt, key = CONF / "cert.pem", CONF / "key.pem"
    if not crt.exists():
        san = ",".join([f"IP:{h}" if h[0].isdigit() else f"DNS:{h}" for h in hosts])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-days", "825", "-subj", "/CN=iphone-mic",
                        "-addext", f"subjectAltName={san}",
                        "-keyout", str(key), "-out", str(crt)],
                       check=True, capture_output=True)
        os.chmod(key, 0o600)
    return crt, key


def local_ips():
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    return [ip for ip in out if not ip.startswith("172.") and ":" not in ip] or ["127.0.0.1"]


class Audio:
    """Owns the virtual device and the pw-cat writer."""

    def __init__(self):
        self.loop = subprocess.Popen([
            "pw-loopback", "--name", "iphone-mic",
            "--capture-props",
            f"media.class=Audio/Sink node.name={SINK} node.description=iPhone-Mic-Input "
            "audio.position=[MONO] priority.session=0 priority.driver=0",
            "--playback-props",
            f"media.class=Audio/Source node.name={SOURCE} node.description=iPhone-Mic "
            "audio.position=[MONO]",
        ])
        self.cat = None
        self.lock = threading.Lock()
        time.sleep(1.0)

    def start(self, rate):
        with self.lock:
            self.stop()
            self.cat = subprocess.Popen(
                ["pw-cat", "--playback", "--target", SINK, "--format", "s16",
                 "--rate", str(rate), "--channels", "1", "--latency", "40ms", "-"],
                stdin=subprocess.PIPE)

    def write(self, data):
        with self.lock:
            if self.cat and self.cat.poll() is None:
                try:
                    self.cat.stdin.write(data)
                    self.cat.stdin.flush()
                except BrokenPipeError:
                    pass

    def stop(self):
        if self.cat:
            try:
                self.cat.stdin.close()
            except Exception:
                pass
            self.cat.terminate()
            self.cat = None

    def close(self):
        self.stop()
        self.loop.terminate()


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return buf


def ws_frames(sock):
    """Yield (opcode, payload) for client frames (always masked)."""
    while True:
        b1, b2 = recv_exact(sock, 2)
        op, ln = b1 & 0x0F, b2 & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", recv_exact(sock, 2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", recv_exact(sock, 8))[0]
        mask = recv_exact(sock, 4) if b2 & 0x80 else b"\0\0\0\0"
        data = bytearray(recv_exact(sock, ln))
        for i in range(ln):
            data[i] ^= mask[i & 3]
        yield op, bytes(data)


class Handler(http.server.BaseHTTPRequestHandler):
    audio: Audio = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("[http] " + fmt % a + "\n")

    def do_GET(self):
        if self.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            return self.websocket()
        body = (HERE / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def websocket(self):
        key = self.headers["Sec-WebSocket-Key"]
        acc = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", acc)
        self.end_headers()
        self.wfile.flush()
        sock = self.connection
        print(f"[mic] phone connected from {self.client_address[0]}", flush=True)
        try:
            for op, data in ws_frames(sock):
                if op == 1:  # text: "rate:48000"
                    if data.startswith(b"rate:"):
                        rate = int(data[5:])
                        print(f"[mic] streaming at {rate} Hz", flush=True)
                        self.audio.start(rate)
                elif op == 2:
                    self.audio.write(data)
                elif op == 8:
                    break
                elif op == 9:  # ping -> pong
                    sock.sendall(bytes([0x8A, len(data)]) + data)
        except (ConnectionError, OSError, ssl.SSLError):
            pass
        finally:
            self.audio.stop()
            self.close_connection = True
            print("[mic] phone disconnected", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--set-default", action="store_true",
                    help="make 'iPhone Mic' the default input")
    args = ap.parse_args()

    for tool in ("pw-loopback", "pw-cat", "openssl"):
        if not shutil.which(tool):
            sys.exit(f"missing {tool}")

    ips = local_ips()
    crt, key = ensure_cert(ips + ["localhost", socket.gethostname()])
    audio = Audio()
    Handler.audio = audio
    if args.set_default:
        subprocess.run(["sh", "-c", f"wpctl set-default $(pw-cli ls Node | "
                        f"awk '/^\\tid/{{id=$2}} /node.name = \"{SOURCE}\"/{{print id}}' | tr -d ,)"])

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    srv.daemon_threads = True

    def bye(*_):
        audio.close()
        os._exit(0)
    signal.signal(signal.SIGTERM, bye)

    print("iPhone mic bridge running. On the iPhone (same Wi-Fi), open in Safari:")
    for ip in ips:
        print(f"    https://{ip}:{args.port}/")
    print("Accept the certificate warning, tap Start, allow the microphone.")
    print("Input device on this PC: 'iPhone-Mic'.  Ctrl+C to quit.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bye()


if __name__ == "__main__":
    main()
