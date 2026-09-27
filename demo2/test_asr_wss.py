#!/usr/bin/env python3
"""Test the server-2 public ASR WebSocket endpoint."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import wave
from pathlib import Path
from typing import Any

from test_public_wss import (
    DEFAULT_BASE_URL,
    _require_websocket_client,
    build_ssl_options,
    chunk_pcm,
    make_connection,
    parse_json_frame,
    tls_error_message,
)


DEFAULT_CA_CERT = Path(__file__).with_name("server2-wss.crt")


def load_pcm16(path: Path) -> bytes:
    """Read raw 16 kHz PCM16LE or validate and extract a matching WAV file."""

    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        payload = path.read_bytes()
    else:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16_000
                or audio.getcomptype() != "NONE"
            ):
                raise ValueError("WAV input must be mono, 16-bit PCM, 16000 Hz")
            payload = audio.readframes(audio.getnframes())
    if not payload:
        raise ValueError("audio file is empty")
    if len(payload) % 2:
        raise ValueError("PCM16LE input must contain complete samples")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--pcm",
        "--asr-pcm",
        dest="pcm",
        type=Path,
        required=True,
        help="16 kHz mono PCM16LE file, or a matching WAV file",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("STS2_ASR_LANGUAGE", "ja"),
        help="ASR language hint; use auto for automatic detection",
    )
    parser.add_argument("--chunk-bytes", type=int, default=640)
    return parser


def run_asr(args: argparse.Namespace) -> int:
    if not args.api_key:
        raise ValueError("set STS2_API_KEY or pass --api-key")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.chunk_bytes <= 0 or args.chunk_bytes % 2:
        raise ValueError("--chunk-bytes must be a positive even number")

    payload = load_pcm16(args.pcm)
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/asr/v1/realtime"
    if args.insecure:
        ssl_options: dict[str, Any] = build_ssl_options(ca_cert=None, insecure=True)
    else:
        if not args.ca_cert.is_file():
            raise FileNotFoundError(f"CA certificate not found: {args.ca_cert}")
        ssl_options = build_ssl_options(ca_cert=str(args.ca_cert), insecure=False)

    started = time.perf_counter()
    ws = None
    try:
        ws = make_connection(
            url,
            api_key=args.api_key,
            ssl_options=ssl_options,
            timeout=args.timeout,
        )
        created = parse_json_frame(ws.recv())
        first_frame = time.perf_counter() - started
        if (
            created.get("type") != "session.created"
            or created.get("audio_format") != "pcm_s16le"
            or int(created.get("sample_rate", 0)) != 16_000
            or int(created.get("channels", 0)) != 1
        ):
            raise RuntimeError(f"invalid ASR session format: {created}")

        ws.send(json.dumps({"type": "session.update", "language": args.language}))
        for chunk in chunk_pcm(payload, args.chunk_bytes):
            ws.send(chunk, opcode=_require_websocket_client().ABNF.OPCODE_BINARY)
        ws.send(json.dumps({"type": "input_audio.commit"}))

        while True:
            message = ws.recv()
            if isinstance(message, bytes):
                raise RuntimeError("ASR endpoint returned an unexpected binary frame")
            event = parse_json_frame(message)
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(event.get("message", "ASR server returned an error"))
            if event_type == "transcription.completed":
                text = event.get("text", "")
                if not isinstance(text, str):
                    raise RuntimeError("ASR transcription text is not a string")
                elapsed = time.perf_counter() - started
                print(
                    f"PASS ASR: {url}"
                    f"\nlanguage={event.get('language', 'und')}"
                    f" text={text!r}"
                    f" audio_bytes={len(payload)}"
                    f" audio_seconds={len(payload) / 32_000:.2f}"
                    f" first_frame={first_frame:.3f}s"
                    f" elapsed={elapsed:.3f}s"
                )
                return 0
    except ssl.SSLCertVerificationError as exc:
        raise RuntimeError(tls_error_message(url, str(args.ca_cert))) from exc
    finally:
        if ws is not None:
            ws.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.insecure:
            print("TLS verification: disabled (--insecure)")
        else:
            print(f"TLS verification: enabled with CA {args.ca_cert}")
        return run_asr(args)
    except Exception as exc:
        print(f"FAIL ASR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
