 import argparse
import json
import os
import ssl
import wave

import websocket


parser = argparse.ArgumentParser()
parser.add_argument("--text", default="こんにちは。ミモの音声サービスです。")
args = parser.parse_args()

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

ws = websocket.create_connection(
    f"{base_url}/tts/v1/realtime",
    header=[f"Authorization: Bearer {api_key}"],
    sslopt=ssl_options,
    timeout=300,
)

audio = bytearray()
sample_rate = 24000

try:
    print("connected:", ws.recv())
    ws.send(
        json.dumps(
            {
                "type": "response.create",
                "text": args.text,
                "voice": "Ono_Anna",
                "language": "ja",
            },
            ensure_ascii=False,
        )
    )

    while True:
        message = ws.recv()
        if isinstance(message, bytes):
            audio.extend(message)
            continue

        print(message)
        event = json.loads(message)
        if event.get("type") == "response.audio.started":
            sample_rate = event.get("sample_rate", sample_rate)
        if event.get("type") == "response.completed":
            break
finally:
    ws.close()

with wave.open("tts-wss-test.wav", "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(sample_rate)
    output.writeframes(audio)

print(f"saved tts-wss-test.wav: {len(audio)} bytes")
