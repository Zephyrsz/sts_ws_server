import json
import os
import ssl
import threading
import time
import wave

import websocket


base_url = os.environ.get(
    "MIMO_WSS_BASE_URL",
    "wss://47.87.138.166:18443",
).rstrip("/")
api_key = os.environ["MIMO_API_KEY"]
ca_cert = os.getenv("MIMO_CA_CERT")
ssl_options = (
    {"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": ca_cert}
    if ca_cert
    else {"cert_reqs": ssl.CERT_NONE}
)

text_chunks = [
    "こんにちは。",
    "これはコージーボイス・スリーを使った",
    "日本語音声合成テストです。",
]

ws = websocket.create_connection(
    f"{base_url}/tts/cosyvoice",
    header=[f"Authorization: Bearer {api_key}"],
    sslopt=ssl_options,
    timeout=300,
)

audio = bytearray()
state = {"start": time.perf_counter(), "end_sent": None, "first_audio": None}
sample_rate = 24000
channels = 1

try:
    ws.send(json.dumps({"type": "start"}))
    audio_format = json.loads(ws.recv())
    print(audio_format)
    sample_rate = int(audio_format["sample_rate"])
    channels = int(audio_format["channels"])

    def send_text():
        for text in text_chunks:
            ws.send(json.dumps({"type": "text", "text": text}, ensure_ascii=False))
            print("sent:", text, flush=True)
            time.sleep(1.0)
        state["end_sent"] = time.perf_counter()
        ws.send(json.dumps({"type": "end"}))
        print("sent: end", flush=True)

    sender = threading.Thread(target=send_text, daemon=True)
    sender.start()

    while True:
        message = ws.recv()
        now = time.perf_counter()

        if isinstance(message, bytes):
            if state["first_audio"] is None:
                state["first_audio"] = now
            audio.extend(message)
            print(f"received audio bytes={len(message)}", flush=True)
            continue

        event = json.loads(message)
        print(event, flush=True)
        if event.get("type") == "done":
            break
        if event.get("type") == "error":
            raise RuntimeError(event.get("message", "server error"))
finally:
    ws.close()

with wave.open("cosyvoice3-wss-test.wav", "wb") as output:
    output.setnchannels(channels)
    output.setsampwidth(2)
    output.setframerate(sample_rate)
    output.writeframes(audio)

before_end = (
    state["first_audio"] is not None
    and state["end_sent"] is not None
    and state["first_audio"] < state["end_sent"]
)
latency = state["first_audio"] - state["start"] if state["first_audio"] else None
print(f"first audio latency: {latency:.3f}s")
print(f"first audio before end sent: {before_end}")
print(f"saved cosyvoice3-wss-test.wav: {len(audio)} bytes")
