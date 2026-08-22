"""Asterisk AudioSocket server — teensyvad's telephony front door.

Run it:
    .venv/bin/python scripts/audiosocket_server.py            # port 9092
    TEENSYVAD_MODEL=models/teensy-v1.npz .venv/bin/python scripts/audiosocket_server.py --port 9092

Then in Asterisk's extensions.conf:

    [teensyvad]
    ext => 500,1,NoOp(teensyvad)
      same => n,Answer()
      same => n,AudioSocket(127.0.0.1:9092,call-${UNIQUEID})
      same => n,Hangup()

(Asterisk ≥ 18, `chan_audiosocket` / app_audiosocket loaded — usually
default.  Every call becomes one TCP connection to this server.)

Protocol (per Asterisk's AudioSocket spec): every message is
    ┌──────────┬────────────┬─────────────┐
    │ type u8  │ length u16 │ payload     │   length is BIG-endian
    └──────────┴────────────┴─────────────┘
    type 0x01 → the call's UUID, once, right after connect
    type 0x03 → audio: 8 kHz, 16-bit signed LE mono ("slin"), 20 ms = 320 B
    type 0x04 → hangup
We feed each 0x03 payload straight into StreamingVAD and log the events.
"""

from __future__ import annotations

import argparse
import socket
import socketserver
import sys
import threading
import time
import uuid as uuidlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.streaming import StreamingVAD  # noqa: E402

HOST = "0.0.0.0"
T_UUID, T_AUDIO, T_HANGUP, T_ERROR = 0x01, 0x03, 0x04, 0xFF


def read_exactly(conn: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes; None on EOF/error."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class VADHandler(socketserver.BaseRequestHandler):
    """One connected Asterisk call = one handler instance."""

    def handle(self) -> None:
        peer = self.client_address[0]
        vad = StreamingVAD(self.server.model_path)
        call_id = "?"
        t0 = time.time()
        n_frames = 0
        print(f"[+] connection from {peer}", flush=True)

        while True:
            hdr = read_exactly(self.request, 3)
            if hdr is None:
                break
            mtype = hdr[0]
            mlen = int.from_bytes(hdr[1:3], "big")
            payload = read_exactly(self.request, mlen) if mlen else b""
            if payload is None:
                break

            if mtype == T_UUID:
                try:
                    call_id = str(uuidlib.UUID(bytes=payload))
                except ValueError:
                    call_id = payload.decode(errors="replace")
                print(f"    call {call_id}: VAD armed "
                      f"({vad.sr} Hz, thr {vad.thr_hi:.2f}/{vad.thr_lo:.2f})", flush=True)

            elif mtype == T_AUDIO:
                n_frames += 1
                for ev in vad.feed(payload):          # 320 B = 20 ms slin
                    wall = time.time() - t0
                    print(f"    [{wall:8.2f}s] {ev.type}  "
                          f"(stream t={ev.t:.2f}s, P={vad.last_prob:.2f})", flush=True)

            elif mtype == T_HANGUP:
                break
            elif mtype == T_ERROR:
                print(f"    ! audiosocket error code {payload!r}", flush=True)
                break

        dur = n_frames * 0.02
        print(f"[-] call {call_id} ended: {dur:5.1f}s audio, "
              f"{vad.speech_seconds:5.1f}s speech ({vad.speech_seconds/max(dur,1e-9)*100:.0f}%)",
              flush=True)


class VADServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=9092)
    ap.add_argument("--model", type=Path, default=None,
                    help="model .npz (default: auto-discover, see TEENSYVAD_MODEL)")
    args = ap.parse_args()

    VADServer.model_path = args.model  # None → StreamingVAD auto-discovers
    srv = VADServer((args.host, args.port), VADHandler)
    print(f"teensyvad AudioSocket server listening on {args.host}:{args.port}")
    print("dialplan:  same => n,AudioSocket(127.0.0.1:9092,call-${UNIQUEID})")
    print("Ctrl-C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        srv.shutdown()


if __name__ == "__main__":
    main()
