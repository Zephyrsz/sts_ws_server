#!/usr/bin/env python3
"""Smoke-test all public WSS endpoints exposed by the Mimo gateway.

The script intentionally prints protocol metadata and counters instead of full
server responses.  It is safe to use in CI logs as long as the API key is
provided through ``MIMO_API_KEY`` or ``--api-key``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import ssl
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_WEBSOCKET_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import websocket
except ModuleNotFoundError as exc:
    if exc.name != "websocket":
        raise
    websocket = None  # type: ignore[assignment]
    _WEBSOCKET_IMPORT_ERROR = exc


def _require_websocket_client() -> Any:
    """Return websocket-client or explain how to repair the active interpreter."""

    python = shlex.quote(sys.executable)
    if _WEBSOCKET_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing dependency 'websocket-client' for this Python interpreter.\n"
            f"Interpreter: {sys.executable}\n"
            f"Install with: uv pip install --python {python} 'websocket-client>=1.8,<2'"
        ) from None
    if not hasattr(websocket, "create_connection") or not hasattr(websocket, "ABNF"):
        module_path = getattr(websocket, "__file__", "unknown")
        raise SystemExit(
            "The imported 'websocket' module is not provided by websocket-client.\n"
            f"Interpreter: {sys.executable}\n"
            f"Imported module: {module_path}\n"
            "A conflicting PyPI package named 'websocket' may be installed. Remove that "
            "package from this environment, then reinstall websocket-client:\n"
            f"uv pip uninstall --python {python} websocket\n"
            f"uv pip install --python {python} --reinstall 'websocket-client>=1.8,<2'"
        )
    return websocket


DEFAULT_BASE_URL = "wss://47.87.138.166:18443"
DEFAULT_COSYVOICE_TEXT = (
    "こんにちは。",
    "これはコージーボイス・スリーを使った",
    "日本語音声合成テストです。",
)


@dataclass
class CaseResult:
    name: str
    url: str
    passed: bool = False
    connect_seconds: float | None = None
    first_frame_seconds: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def chunk_pcm(data: bytes, chunk_bytes: int = 640) -> Iterable[bytes]:
    """Yield complete PCM16 chunks without changing their byte representation."""

    if chunk_bytes <= 0 or chunk_bytes % 2:
        raise ValueError("chunk_bytes must be a positive even number")
    if len(data) % 2:
        raise ValueError("PCM16 payload must contain an even number of bytes")
    for offset in range(0, len(data), chunk_bytes):
        yield data[offset : offset + chunk_bytes]


def parse_json_frame(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        raise ValueError("expected a JSON object text frame")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError("expected a JSON object text frame") from exc
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object text frame")
    return payload


def parse_llm_frame(message: str | bytes) -> tuple[bool, str]:
    """Return ``(is_done, content_delta)`` for one LLM WebSocket frame."""

    if isinstance(message, bytes):
        raise ValueError("LLM endpoint must return text frames")
    if message == "[DONE]":
        return True, ""
    payload = parse_json_frame(message)
    if "error" in payload:
        raise RuntimeError("LLM server returned an error")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM frame has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("LLM frame choice is not an object")
    delta = choice.get("delta")
    if delta is None:
        return False, ""
    if not isinstance(delta, dict):
        raise ValueError("LLM frame delta is not an object")
    content = delta.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("LLM content delta is not a string")
    return False, content


def build_ssl_options(*, ca_cert: str | None, insecure: bool) -> dict[str, Any]:
    if insecure:
        return {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
    options: dict[str, Any] = {"cert_reqs": ssl.CERT_REQUIRED}
    if ca_cert:
        options["ca_certs"] = ca_cert
    return options


def tls_error_message(url: str, ca_cert: str | None) -> str:
    return (
        f"TLS certificate verification failed for {url}. "
        f"MIMO_CA_CERT={ca_cert or '<system CA store>'} may not be the public gateway certificate. "
        "Use the CA/certificate served by the 18443 gateway, or for a temporary diagnostic run "
        "with --insecure."
    )


def make_connection(
    url: str,
    *,
    api_key: str,
    ssl_options: dict[str, Any],
    timeout: float,
) -> websocket.WebSocket:
    client = _require_websocket_client()
    return client.create_connection(
        url,
        header=[f"Authorization: Bearer {api_key}"],
        sslopt=ssl_options,
        timeout=timeout,
    )


def save_wav(path: Path, audio: bytes, *, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio)


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def run_asr(
    *,
    base_url: str,
    api_key: str,
    ssl_options: dict[str, Any],
    timeout: float,
    pcm_path: Path | None,
    language: str,
    chunk_bytes: int,
) -> CaseResult:
    url = f"{base_url}/asr/v1/realtime"
    result = CaseResult("asr", url)
    started = time.perf_counter()
    try:
        if pcm_path is None:
            raise ValueError("--asr-pcm is required unless --skip-asr is used")
        if not pcm_path.is_file():
            raise FileNotFoundError(f"PCM file not found: {pcm_path}")
        ws = make_connection(
            url, api_key=api_key, ssl_options=ssl_options, timeout=timeout
        )
        result.connect_seconds = _elapsed(started)
        try:
            event = parse_json_frame(ws.recv())
            result.first_frame_seconds = _elapsed(started)
            if (
                event.get("type") != "session.created"
                or event.get("audio_format") != "pcm_s16le"
                or int(event.get("sample_rate", 0)) != 16_000
                or int(event.get("channels", 0)) != 1
            ):
                raise RuntimeError("invalid ASR session format")
            ws.send(json.dumps({"type": "session.update", "language": language}))

            frames = total_bytes = 0
            with pcm_path.open("rb") as audio:
                payload = audio.read()
            for chunk in chunk_pcm(payload, chunk_bytes):
                binary_opcode = getattr(getattr(websocket, "ABNF", None), "OPCODE_BINARY", 2)
                ws.send(chunk, opcode=binary_opcode)
                frames += 1
                total_bytes += len(chunk)
            if not total_bytes:
                raise ValueError("ASR PCM file is empty")
            ws.send(json.dumps({"type": "input_audio.commit"}))

            while True:
                event = parse_json_frame(ws.recv())
                if event.get("type") == "error":
                    raise RuntimeError("ASR server returned an error")
                if event.get("type") == "transcription.completed":
                    text = event.get("text", "")
                    if not isinstance(text, str):
                        raise ValueError("ASR transcription text is not a string")
                    result.metrics = {
                        "audio_frames": frames,
                        "audio_bytes": total_bytes,
                        "audio_seconds": round(total_bytes / 32_000, 2),
                        "text_chars": len(text),
                    }
                    result.passed = True
                    break
        finally:
            ws.close()
    except ssl.SSLCertVerificationError:
        result.error = tls_error_message(url, ssl_options.get("ca_certs"))
    except Exception as exc:
        result.error = str(exc)
    return result


def run_qwen_tts(
    *,
    base_url: str,
    api_key: str,
    ssl_options: dict[str, Any],
    timeout: float,
    text: str,
    voice: str,
    language: str,
    output_path: Path | None,
) -> CaseResult:
    url = f"{base_url}/tts/v1/realtime"
    result = CaseResult("qwen-tts", url)
    started = time.perf_counter()
    audio = bytearray()
    sample_rate = 24_000
    channels = 1
    try:
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        ws = make_connection(
            url, api_key=api_key, ssl_options=ssl_options, timeout=timeout
        )
        result.connect_seconds = _elapsed(started)
        try:
            ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "text": text,
                        "voice": voice,
                        "language": language,
                    },
                    ensure_ascii=False,
                )
            )
            audio_frames = text_frames = 0
            saw_started = saw_completed = False
            while True:
                message = ws.recv()
                if result.first_frame_seconds is None:
                    result.first_frame_seconds = _elapsed(started)
                if isinstance(message, bytes):
                    audio.extend(message)
                    audio_frames += 1
                    continue
                event = parse_json_frame(message)
                event_type = event.get("type")
                text_frames += 1
                if event_type == "response.audio.started":
                    saw_started = True
                    if event.get("format") != "pcm_s16le":
                        raise RuntimeError("invalid Qwen TTS audio format")
                    sample_rate = int(event.get("sample_rate", sample_rate))
                    channels = int(event.get("channels", channels))
                elif event_type == "response.completed":
                    saw_completed = True
                    break
                elif event_type == "error":
                    raise RuntimeError("Qwen TTS server returned an error")
            if not saw_started or not saw_completed:
                raise RuntimeError("Qwen TTS response was incomplete")
            if not audio:
                raise RuntimeError("Qwen TTS returned no audio frames")
            if output_path:
                save_wav(output_path, bytes(audio), sample_rate=sample_rate, channels=channels)
            result.metrics = {
                "audio_frames": audio_frames,
                "audio_bytes": len(audio),
                "text_frames": text_frames,
                "sample_rate": sample_rate,
            }
            result.passed = True
        finally:
            ws.close()
    except ssl.SSLCertVerificationError:
        result.error = tls_error_message(url, ssl_options.get("ca_certs"))
    except Exception as exc:
        result.error = str(exc)
    return result


def run_cosyvoice(
    *,
    base_url: str,
    api_key: str,
    ssl_options: dict[str, Any],
    timeout: float,
    text_chunks: tuple[str, ...],
    output_path: Path | None,
) -> CaseResult:
    url = f"{base_url}/tts/cosyvoice"
    result = CaseResult("cosyvoice", url)
    started = time.perf_counter()
    audio = bytearray()
    sample_rate = 24_000
    channels = 1
    first_audio_seconds: float | None = None
    try:
        if not text_chunks or any(not item.strip() for item in text_chunks):
            raise ValueError("CosyVoice text chunks must not be empty")
        ws = make_connection(
            url, api_key=api_key, ssl_options=ssl_options, timeout=timeout
        )
        result.connect_seconds = _elapsed(started)
        try:
            ws.send(json.dumps({"type": "start"}))
            event = parse_json_frame(ws.recv())
            result.first_frame_seconds = _elapsed(started)
            if (
                event.get("type") != "audio_format"
                or event.get("format") != "pcm_s16le"
            ):
                raise RuntimeError("invalid CosyVoice audio format")
            sample_rate = int(event["sample_rate"])
            channels = int(event["channels"])
            for text in text_chunks:
                ws.send(json.dumps({"type": "text", "text": text}, ensure_ascii=False))
            ws.send(json.dumps({"type": "end"}))

            audio_frames = 0
            saw_done = False
            while True:
                message = ws.recv()
                if isinstance(message, bytes):
                    if first_audio_seconds is None:
                        first_audio_seconds = _elapsed(started)
                    audio.extend(message)
                    audio_frames += 1
                    continue
                event = parse_json_frame(message)
                if event.get("type") == "error":
                    raise RuntimeError("CosyVoice server returned an error")
                if event.get("type") == "done":
                    saw_done = True
                    break
            if not saw_done or not audio:
                raise RuntimeError("CosyVoice response was incomplete")
            if output_path:
                save_wav(output_path, bytes(audio), sample_rate=sample_rate, channels=channels)
            result.metrics = {
                "audio_frames": audio_frames,
                "audio_bytes": len(audio),
                "text_chunks": len(text_chunks),
                "sample_rate": sample_rate,
                "first_audio_seconds": round(first_audio_seconds or 0.0, 3),
            }
            result.passed = True
        finally:
            ws.close()
    except ssl.SSLCertVerificationError:
        result.error = tls_error_message(url, ssl_options.get("ca_certs"))
    except Exception as exc:
        result.error = str(exc)
    return result


def run_llm(
    *,
    base_url: str,
    api_key: str,
    ssl_options: dict[str, Any],
    timeout: float,
    model: str,
    prompt: str,
) -> CaseResult:
    url = f"{base_url}/ws/v1/chat/completions"
    result = CaseResult("llm", url)
    started = time.perf_counter()
    try:
        ws = make_connection(
            url, api_key=api_key, ssl_options=ssl_options, timeout=timeout
        )
        result.connect_seconds = _elapsed(started)
        try:
            ws.send(
                json.dumps(
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Do not explain or reason. Return only the requested token.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 128,
                        "temperature": 0,
                        "stream": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    ensure_ascii=False,
                )
            )
            frames = content_chars = 0
            saw_done = False
            while True:
                message = ws.recv()
                if result.first_frame_seconds is None:
                    result.first_frame_seconds = _elapsed(started)
                done, content = parse_llm_frame(message)
                frames += 1
                content_chars += len(content)
                if done:
                    saw_done = True
                    break
            if not saw_done or content_chars == 0:
                raise RuntimeError("LLM response was incomplete or had no content")
            result.metrics = {
                "text_frames": frames,
                "content_chars": content_chars,
                "done": True,
            }
            result.passed = True
        finally:
            ws.close()
    except ssl.SSLCertVerificationError:
        result.error = tls_error_message(url, ssl_options.get("ca_certs"))
    except Exception as exc:
        result.error = str(exc)
    return result


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=os.getenv("MIMO_WSS_BASE_URL", DEFAULT_BASE_URL),
        help="WSS gateway base URL (default: MIMO_WSS_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("MIMO_API_KEY"),
        help="Bearer API key (default: MIMO_API_KEY)",
    )
    parser.add_argument(
        "--ca-cert",
        default=os.getenv("MIMO_CA_CERT"),
        help="CA certificate for the gateway (default: MIMO_CA_CERT)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate and hostname verification",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="WebSocket connect/read timeout in seconds",
    )
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-qwen-tts", action="store_true")
    parser.add_argument("--skip-cosyvoice", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_arguments(parser)
    parser.add_argument("--asr-pcm", type=Path, help="16 kHz mono PCM16LE file")
    parser.add_argument("--asr-language", default="ja")
    parser.add_argument("--asr-chunk-bytes", type=int, default=640)
    parser.add_argument("--text", default="こんにちは。ミモの音声サービスです。")
    parser.add_argument("--voice", default="Ono_Anna")
    parser.add_argument(
        "--language",
        default="Japanese",
        help="Qwen TTS language enum (default: Japanese; ASR uses --asr-language)",
    )
    parser.add_argument("--cosyvoice-text", action="append", dest="cosyvoice_text")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="save returned TTS audio as WAV files in this directory",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("MIMO_LLM_MODEL", "Ornith-1.5-35B-A3B"),
    )
    parser.add_argument("--llm-prompt", default="Reply with the single word OK.")
    return parser


def print_result(result: CaseResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    timing = f"connect={result.connect_seconds:.3f}s" if result.connect_seconds is not None else "connect=-"
    if result.first_frame_seconds is not None:
        timing += f", first_frame={result.first_frame_seconds:.3f}s"
    detail = ", ".join(f"{key}={value}" for key, value in result.metrics.items())
    if result.error:
        detail = f"error={result.error}"
    print(f"[{status}] {result.name}: {result.url} ({timing}{', ' if detail else ''}{detail})")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _require_websocket_client()
    if not args.api_key:
        parser.error("MIMO_API_KEY or --api-key is required")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    base_url = args.base_url.rstrip("/")
    ssl_options = build_ssl_options(ca_cert=args.ca_cert, insecure=args.insecure)
    if args.insecure:
        print("TLS verification: disabled (--insecure)")
    elif args.ca_cert:
        print(f"TLS verification: enabled with CA {args.ca_cert}")
    else:
        print("TLS verification: enabled with system CA store")

    output_dir: Path | None = args.output_dir
    cosyvoice_text = tuple(args.cosyvoice_text or DEFAULT_COSYVOICE_TEXT)
    results: list[CaseResult] = []
    if not args.skip_asr:
        results.append(
            run_asr(
                base_url=base_url,
                api_key=args.api_key,
                ssl_options=ssl_options,
                timeout=args.timeout,
                pcm_path=args.asr_pcm,
                language=args.asr_language,
                chunk_bytes=args.asr_chunk_bytes,
            )
        )
    if not args.skip_qwen_tts:
        results.append(
            run_qwen_tts(
                base_url=base_url,
                api_key=args.api_key,
                ssl_options=ssl_options,
                timeout=args.timeout,
                text=args.text,
                voice=args.voice,
                language=args.language,
                output_path=output_dir / "qwen-tts.wav" if output_dir else None,
            )
        )
    if not args.skip_cosyvoice:
        results.append(
            run_cosyvoice(
                base_url=base_url,
                api_key=args.api_key,
                ssl_options=ssl_options,
                timeout=args.timeout,
                text_chunks=cosyvoice_text,
                output_path=output_dir / "cosyvoice.wav" if output_dir else None,
            )
        )
    if not args.skip_llm:
        results.append(
            run_llm(
                base_url=base_url,
                api_key=args.api_key,
                ssl_options=ssl_options,
                timeout=args.timeout,
                model=args.llm_model,
                prompt=args.llm_prompt,
            )
        )

    if not results:
        parser.error("at least one service must be enabled")
    for result in results:
        print_result(result)
    passed = sum(result.passed for result in results)
    print(f"Summary: {passed}/{len(results)} services passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
