import json
import os
import ssl
import sys

import websocket


base_url = os.environ.get(
    "MIMO_WSS_BASE_URL",
    "wss://47.87.138.166:18443",
).rstrip("/")
api_key = os.environ["MIMO_API_KEY"]
pcm_path = sys.argv[1] if len(sys.argv) > 1 else "test-ja.pcm"
ca_cert = os.getenv("MIMO_CA_CERT")
ssl_options = (
    {"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": ca_cert}
    if ca_cert
    else {"cert_reqs": ssl.CERT_NONE}
)

ws = websocket.create_connection(
    f"{base_url}/asr/v1/realtime",
    header=[f"Authorization: Bearer {api_key}"],
    sslopt=ssl_options,
    timeout=180,
)

try:
    print("connected:", ws.recv())
    ws.send(json.dumps({"type": "session.update", "language": "ja"}))

    frames = total_bytes = 0
    with open(pcm_path, "rb") as audio:
        while chunk := audio.read(640):
            ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
            frames += 1
            total_bytes += len(chunk)

    print(f"sent frames={frames}, bytes={total_bytes}, seconds={total_bytes / 32000:.2f}")
    ws.send(json.dumps({"type": "input_audio.commit"}))

    while True:
        message = ws.recv()
        print(message)
        if isinstance(message, str):
            event = json.loads(message)
            if event.get("type") == "transcription.completed":
                break
finally:
    ws.close()
