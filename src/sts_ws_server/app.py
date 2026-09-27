from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from sts_ws_server.backends import (
    ASRBackend,
    ModelBackendError,
    Transcript,
    TTSBackend,
    create_asr_backend,
    create_tts_backend,
)
from sts_ws_server.config import Settings
from sts_ws_server.vad import EnergyVAD, SileroONNXVAD

logger = logging.getLogger(__name__)


def _enabled(service_name: str, settings: Settings) -> bool:
    return settings.service_name in {"all", service_name}


def _is_authorized(websocket: WebSocket, settings: Settings) -> bool:
    authorization = websocket.headers.get("authorization", "")
    expected = f"Bearer {settings.api_key}"
    return bool(settings.api_key) and hmac.compare_digest(authorization, expected)


async def _accept_or_close(websocket: WebSocket, settings: Settings, service_name: str) -> bool:
    if not _enabled(service_name, settings):
        await websocket.close(code=4404, reason="service is not enabled")
        return False
    if not _is_authorized(websocket, settings):
        await websocket.close(code=4401, reason="valid bearer token required")
        return False
    await websocket.accept()
    return True


async def _error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


def _parse_text_message(message: dict[str, Any]) -> dict[str, Any] | None:
    raw_text = message.get("text")
    if not isinstance(raw_text, str):
        return None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _string_field(payload: dict[str, Any], name: str, *, required: bool = False) -> str | None:
    value = payload.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"'{name}' must be a non-empty string")
    return value


def _pcm_frames(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    async def aligned() -> AsyncIterator[bytes]:
        pending = b""
        async for chunk in chunks:
            data = pending + chunk
            usable = len(data) - len(data) % 2
            if usable:
                yield data[:usable]
            pending = data[usable:]
        if pending:
            raise ModelBackendError("TTS backend returned an incomplete PCM16 sample")

    return aligned()


async def _stream_tts(
    websocket: WebSocket,
    backend: TTSBackend,
    limiter: asyncio.Semaphore,
    text: str,
    voice: str | None,
    language: str | None,
) -> bool:
    sent_audio = False
    async with limiter:
        chunks = backend.stream_audio(text, voice, language)
        async for chunk in _pcm_frames(chunks):
            if chunk:
                await websocket.send_bytes(chunk)
                sent_audio = True
    if not sent_audio:
        raise ModelBackendError("TTS backend returned no audio")
    return sent_audio


def create_app(
    settings: Settings,
    *,
    asr_backend: ASRBackend | None = None,
    qwen_tts_backend: TTSBackend | None = None,
    cosyvoice_tts_backend: TTSBackend | None = None,
    kokoro_tts_backend: TTSBackend | None = None,
) -> FastAPI:
    active_asr = asr_backend
    active_qwen = qwen_tts_backend
    active_cosyvoice = cosyvoice_tts_backend
    active_kokoro = kokoro_tts_backend
    if _enabled("asr", settings) and active_asr is None:
        active_asr = create_asr_backend(settings)
    if _enabled("qwen-tts", settings) and active_qwen is None:
        active_qwen = create_tts_backend(settings, "qwen-tts")
    if _enabled("cosyvoice", settings) and active_cosyvoice is None:
        active_cosyvoice = create_tts_backend(settings, "cosyvoice")
    if _enabled("kokoro", settings) and active_kokoro is None:
        active_kokoro = create_tts_backend(settings, "kokoro")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            closed: set[int] = set()
            for backend in (active_asr, active_qwen, active_cosyvoice, active_kokoro):
                if backend is None or id(backend) in closed:
                    continue
                closed.add(id(backend))
                close = getattr(backend, "aclose", None)
                if callable(close):
                    try:
                        await close()
                    except Exception:
                        logger.exception("Failed to close model backend resources")

    app = FastAPI(
        title="Speech Model WebSocket Services",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.asr_backend = active_asr
    app.state.qwen_tts_backend = active_qwen
    app.state.cosyvoice_tts_backend = active_cosyvoice
    app.state.kokoro_tts_backend = active_kokoro
    app.state.inference_limiter = asyncio.Semaphore(settings.max_concurrent_requests)
    if settings.vad_backend == "energy":
        vad = EnergyVAD(
            sample_rate=16_000,
            frame_ms=settings.vad_frame_ms,
            threshold=settings.vad_rms_threshold,
            padding_ms=settings.vad_padding_ms,
        )
    elif settings.vad_backend == "silero":
        if not settings.vad_model_path:
            raise ValueError("VAD_BACKEND=silero requires VAD_MODEL_PATH")
        vad = SileroONNXVAD(
            settings.vad_model_path,
            threshold=settings.vad_threshold,
            frame_ms=settings.vad_frame_ms,
            padding_ms=settings.vad_padding_ms,
            provider=settings.vad_provider,
        )
    else:
        raise ValueError("VAD_BACKEND must be energy or silero")

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        active = [
            name for name in ("asr", "qwen-tts", "cosyvoice", "kokoro")
            if _enabled(name, settings)
        ]
        loaded = all(
            backend is not None
            for name, backend in (
                ("asr", active_asr),
                ("qwen-tts", active_qwen),
                ("cosyvoice", active_cosyvoice),
                ("kokoro", active_kokoro),
            )
            if name in active
        )
        return {
            "status": "ok" if loaded and settings.api_key else "not_ready",
            "service": settings.service_name,
            "protocols": active,
        }

    @app.websocket("/asr/v1/realtime")
    async def asr_realtime(websocket: WebSocket) -> None:
        if not await _accept_or_close(websocket, settings, "asr"):
            return
        assert active_asr is not None
        await websocket.send_json(
            {
                "type": "session.created",
                "audio_format": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        audio = bytearray()
        language: str | None = None
        max_audio_bytes = 16_000 * 2 * settings.max_audio_seconds
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                binary = message.get("bytes")
                if binary is not None:
                    if len(binary) % 2:
                        await _error(websocket, "audio frames must contain complete PCM16 samples")
                        continue
                    if len(audio) + len(binary) > max_audio_bytes:
                        audio.clear()
                        await _error(websocket, f"audio exceeds {settings.max_audio_seconds} seconds")
                        continue
                    audio.extend(binary)
                    continue

                payload = _parse_text_message(message)
                if payload is None:
                    await _error(websocket, "expected a JSON object message")
                    continue
                message_type = payload.get("type")
                if message_type == "session.update":
                    try:
                        language = _string_field(payload, "language")
                        if language and len(language) > 32:
                            raise ValueError("'language' is too long")
                    except ValueError as exc:
                        await _error(websocket, str(exc))
                    continue
                if message_type != "input_audio.commit":
                    await _error(websocket, "expected session.update or input_audio.commit")
                    continue

                committed = bytes(audio)
                audio.clear()
                try:
                    speech = vad.filter(committed)
                    if speech:
                        async with app.state.inference_limiter:
                            result = await active_asr.transcribe(speech, language)
                    else:
                        result = Transcript(text="", language=language or "und", segments=[])
                except ModelBackendError as exc:
                    await _error(websocket, str(exc))
                    continue
                except Exception:
                    logger.exception("ASR request failed")
                    await _error(websocket, "ASR inference failed")
                    continue

                await websocket.send_json(
                    {
                        "type": "transcription.completed",
                        "text": result.text,
                        "language": result.language or language or "und",
                        "segments": result.segments or [],
                        "is_final": True,
                    }
                )
        except WebSocketDisconnect:
            return

    @app.websocket("/tts/v1/realtime")
    async def qwen_tts_realtime(websocket: WebSocket) -> None:
        if not await _accept_or_close(websocket, settings, "qwen-tts"):
            return
        assert active_qwen is not None
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                payload = _parse_text_message(message)
                if payload is None:
                    await _error(websocket, "expected a JSON object message")
                    continue
                if payload.get("type") != "response.create":
                    await _error(websocket, "expected response.create")
                    continue
                try:
                    text = _string_field(payload, "text", required=True)
                    voice = _string_field(payload, "voice") or settings.qwen_tts_voice
                    language = _string_field(payload, "language")
                    if len(text or "") > settings.max_text_chars:
                        raise ValueError(f"text exceeds {settings.max_text_chars} characters")
                    await websocket.send_json(
                        {
                            "type": "response.audio.started",
                            "format": "pcm_s16le",
                            "sample_rate": settings.qwen_tts_sample_rate,
                            "channels": 1,
                        }
                    )
                    await _stream_tts(
                        websocket,
                        active_qwen,
                        app.state.inference_limiter,
                        text or "",
                        voice,
                        language,
                    )
                    await websocket.send_json({"type": "response.completed"})
                except ValueError as exc:
                    await _error(websocket, str(exc))
                except WebSocketDisconnect:
                    return
                except ModelBackendError as exc:
                    await _error(websocket, str(exc))
                except Exception:
                    logger.exception("Qwen-compatible TTS request failed")
                    await _error(websocket, "TTS inference failed")
        except WebSocketDisconnect:
            return

    @app.websocket("/tts/cosyvoice")
    async def cosyvoice_realtime(websocket: WebSocket) -> None:
        if not await _accept_or_close(websocket, settings, "cosyvoice"):
            return
        assert active_cosyvoice is not None
        started = False
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                payload = _parse_text_message(message)
                if payload is None:
                    await _error(websocket, "expected a JSON object message")
                    continue
                message_type = payload.get("type")
                if message_type == "start" and not started:
                    started = True
                    await websocket.send_json(
                        {
                            "type": "audio_format",
                            "sample_rate": settings.cosyvoice_tts_sample_rate,
                            "channels": 1,
                            "format": "pcm_s16le",
                        }
                    )
                    continue
                if not started:
                    await _error(websocket, "send start before text or end")
                    continue
                if message_type == "text":
                    try:
                        text = _string_field(payload, "text", required=True)
                        if len(text or "") > settings.max_text_chars:
                            raise ValueError(f"text exceeds {settings.max_text_chars} characters")
                        await _stream_tts(
                            websocket,
                            active_cosyvoice,
                            app.state.inference_limiter,
                            text or "",
                            settings.cosyvoice_tts_voice,
                            _string_field(payload, "language"),
                        )
                    except ValueError as exc:
                        await _error(websocket, str(exc))
                    except WebSocketDisconnect:
                        return
                    except ModelBackendError as exc:
                        await _error(websocket, str(exc))
                    except Exception:
                        logger.exception("CosyVoice-compatible TTS request failed")
                        await _error(websocket, "TTS inference failed")
                    continue
                if message_type == "end":
                    await websocket.send_json({"type": "done"})
                    return
                await _error(websocket, "expected text or end")
        except WebSocketDisconnect:
            return

    @app.websocket("/tts/kokoro")
    async def kokoro_realtime(websocket: WebSocket) -> None:
        if not await _accept_or_close(websocket, settings, "kokoro"):
            return
        assert active_kokoro is not None
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                payload = _parse_text_message(message)
                if payload is None:
                    await _error(websocket, "expected a JSON object message")
                    continue
                if payload.get("type") != "response.create":
                    await _error(websocket, "expected response.create")
                    continue
                try:
                    text = _string_field(payload, "text", required=True)
                    voice = _string_field(payload, "voice") or settings.kokoro_tts_voice
                    language = _string_field(payload, "language") or settings.kokoro_tts_language
                    if len(text or "") > settings.kokoro_tts.max_text_chars:
                        raise ValueError(
                            f"text exceeds {settings.kokoro_tts.max_text_chars} characters"
                        )
                    await websocket.send_json(
                        {
                            "type": "response.audio.started",
                            "format": "pcm_s16le",
                            "sample_rate": settings.kokoro_tts_sample_rate,
                            "channels": 1,
                        }
                    )
                    await _stream_tts(
                        websocket,
                        active_kokoro,
                        app.state.inference_limiter,
                        text or "",
                        voice,
                        language,
                    )
                    await websocket.send_json({"type": "response.completed"})
                except ValueError as exc:
                    await _error(websocket, str(exc))
                except WebSocketDisconnect:
                    return
                except ModelBackendError as exc:
                    await _error(websocket, str(exc))
                except Exception:
                    logger.exception("Kokoro TTS request failed")
                    await _error(websocket, "TTS inference failed")
        except WebSocketDisconnect:
            return

    return app


app = create_app(Settings.from_env())
