#!/usr/bin/env python3
"""Shared helpers for the server-2 public Japanese TTS smoke tests."""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import wave
from pathlib import Path
from typing import Any

try:
    import websocket
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the CLI
    if exc.name == "websocket":
        websocket = None  # type: ignore[assignment]
    else:
        raise


DEFAULT_BASE_URL = "wss://8.211.122.183:18443"
DEFAULT_CA_CERT = Path(__file__).with_name("server2-wss.crt")
DEFAULT_TEXT = "こんにちは。今日はいい天気ですね。"


def require_websocket_client() -> Any:
    """Return websocket-client and explain the common conflicting package issue."""

    if websocket is None or not hasattr(websocket, "create_connection"):
        module_path = getattr(websocket, "__file__", "not importable")
        raise RuntimeError(
            "websocket-client is required (the import must expose create_connection); "
            f"interpreter={sys.executable}; imported={module_path}. "
            "Install with: python -m pip install -U 'websocket-client>=1.8,<2' "
            "and remove the conflicting package named 'websocket' if present."
        )
    return websocket


def add_common_arguments(parser: Any) -> None:
    parser.add_argument(
        "--base-url",
        default=os.getenv("STS2_WSS_BASE_URL", DEFAULT_BASE_URL),
        help="WSS gateway base URL (default: STS2_WSS_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("STS2_API_KEY"),
        help="Bearer API key (default: STS2_API_KEY)",
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        default=Path(os.getenv("STS2_CA_CERT", str(DEFAULT_CA_CERT))),
        help="Gateway certificate/CA (default: demo2/server2-wss.crt)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for temporary diagnostics",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path)


def ssl_options(ca_cert: Path, insecure: bool) -> dict[str, Any]:
    if insecure:
        return {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
    if not ca_cert.is_file():
        raise FileNotFoundError(f"CA certificate not found: {ca_cert}")
    return {"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": str(ca_cert)}


def connect(url: str, api_key: str, options: dict[str, Any], timeout: float) -> Any:
    client = require_websocket_client()
    return client.create_connection(
        url,
        timeout=timeout,
        header=[f"Authorization: Bearer {api_key}"],
        sslopt=options,
    )


def json_event(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        raise RuntimeError("expected a JSON text frame")
    event = json.loads(message)
    if not isinstance(event, dict):
        raise RuntimeError("expected a JSON object event")
    return event


def receive_audio(
    ws: Any,
    *,
    started_at: float,
    service_name: str,
    request: dict[str, Any],
) -> tuple[bytes, int, int, float | None, dict[str, Any]]:
    """Send one response.create request and collect PCM16LE until completed."""

    ws.send(json.dumps(request, ensure_ascii=False))
    audio = bytearray()
    sample_rate = 24_000
    channels = 1
    first_audio_ms: float | None = None
    saw_started = saw_completed = False
    events: dict[str, Any] = {"text_frames": 0, "audio_frames": 0}
    while True:
        message = ws.recv()
        if isinstance(message, bytes):
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - started_at) * 1000
            audio.extend(message)
            events["audio_frames"] += 1
            continue
        event = json_event(message)
        events["text_frames"] += 1
        event_type = event.get("type")
        if event_type == "response.audio.started":
            if event.get("format") != "pcm_s16le":
                raise RuntimeError(f"{service_name}: unsupported audio format: {event}")
            sample_rate = int(event.get("sample_rate", sample_rate))
            channels = int(event.get("channels", channels))
            saw_started = True
        elif event_type == "response.completed":
            saw_completed = True
            break
        elif event_type == "error":
            raise RuntimeError(event.get("message", f"{service_name} request failed"))
    if not saw_started or not saw_completed:
        raise RuntimeError(f"{service_name}: incomplete response")
    if not audio or len(audio) % 2:
        raise RuntimeError(f"{service_name}: no complete PCM16LE audio returned")
    return bytes(audio), sample_rate, channels, first_audio_ms, events


def write_wav(path: Path, audio: bytes, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio)


def run_response_create(
    *,
    service_name: str,
    endpoint: str,
    base_url: str,
    api_key: str,
    ca_cert: Path,
    insecure: bool,
    timeout: float,
    request: dict[str, Any],
    output: Path,
) -> int:
    if not api_key:
        raise ValueError("set STS2_API_KEY or pass --api-key")
    if not request.get("text", "").strip():
        raise ValueError("--text must not be empty")
    if timeout <= 0:
        raise ValueError("--timeout must be positive")

    url = f"{base_url.rstrip('/')}{endpoint}"
    options = ssl_options(ca_cert, insecure)
    print("TLS verification: disabled (--insecure)" if insecure else f"TLS verification: enabled with CA {ca_cert}")
    started_at = time.perf_counter()
    ws = connect(url, api_key, options, timeout)
    try:
        audio, sample_rate, channels, first_audio_ms, events = receive_audio(
            ws,
            started_at=started_at,
            service_name=service_name,
            request=request,
        )
    finally:
        ws.close()
    write_wav(output, audio, sample_rate, channels)
    print(f"PASS {service_name}: {url}")
    print(
        f"sample_rate={sample_rate} channels={channels} pcm_bytes={len(audio)} "
        f"audio_frames={events['audio_frames']} text_frames={events['text_frames']}"
    )
    print(f"first_audio_ms={first_audio_ms:.0f}" if first_audio_ms is not None else "first_audio_ms=none")
    print(f"output={output}")
    return 0
