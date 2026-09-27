from __future__ import annotations

import asyncio
import hmac
import logging
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)


class SarashinaBackendProtocol(Protocol):
    sample_rate: int
    ready: bool
    not_ready_reason: str

    async def synthesize(self, text: str) -> bytes: ...


class SarashinaBackend:
    sample_rate = 24_000

    def __init__(
        self,
        generator,
        *,
        prompt_path: str = "",
        prompt_text: str = "",
        max_tokens: int = 2048,
    ) -> None:
        self.generator = generator
        self.prompt_path = prompt_path
        self.prompt_text = prompt_text.strip()
        self.max_tokens = max_tokens
        self._lock = asyncio.Lock()
        self.ready = True
        self.not_ready_reason = ""
        self.flow_embedding = None
        self.audio_prompt_tokens = None
        self.audio_prompt_feat = None

        if not self.prompt_path or not self.prompt_text:
            if self.prompt_path or self.prompt_text:
                self.ready = False
                self.not_ready_reason = "reference audio and transcript must be configured together"
            return
        if not Path(self.prompt_path).is_file():
            self.ready = False
            self.not_ready_reason = "reference audio file is not available"
            return

        self.flow_embedding = generator._extract_zero_shot_embedding(self.prompt_path)
        self.audio_prompt_tokens = generator._extract_audio_prompt_tokens(self.prompt_path)
        self.audio_prompt_feat = generator._extract_audio_prompt_feat(self.prompt_path)
        self.ready = True
        self.not_ready_reason = ""

    async def synthesize(self, text: str) -> bytes:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize, text)

    def _synthesize(self, text: str) -> bytes:
        wavs = self.generator.generate(
            texts=[text],
            flow_embedding=self.flow_embedding,
            audio_prompt_text=self.prompt_text or None,
            audio_prompt_tokens=self.audio_prompt_tokens,
            audio_prompt_feat=self.audio_prompt_feat,
            audio_prompt_path=self.prompt_path or None,
            gen_kwargs={"max_tokens": self.max_tokens},
        )
        if not wavs:
            raise RuntimeError("Sarashina returned no audio")
        audio = wavs[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Sarashina returned empty audio")
        return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def _build_backend_from_env() -> SarashinaBackend:
    try:
        from sarashina_tts.generate.generate import SarashinaTTSGenerator
    except ImportError as exc:
        raise RuntimeError("Sarashina runtime is not installed in this environment") from exc

    model_path = os.environ.get(
        "SARASHINA_TTS_MODEL_PATH",
        "/app/sts_ws_server/models/tts/sarashina2.2-tts",
    )
    logger.info("Loading Sarashina TTS model from %s", model_path)
    generator = SarashinaTTSGenerator(
        model_dir=model_path,
        device=os.environ.get("SARASHINA_TTS_DEVICE", "cuda:0"),
        watermark=True,
    )
    return SarashinaBackend(
        generator,
        prompt_path=os.environ.get("SARASHINA_REFERENCE_AUDIO", ""),
        prompt_text=os.environ.get("SARASHINA_REFERENCE_TEXT", ""),
        max_tokens=int(os.environ.get("SARASHINA_MAX_TOKENS", "2048")),
    )


def create_app(
    *,
    api_key: str,
    backend: SarashinaBackendProtocol | None,
    chunk_bytes: int = 4096,
    max_text_chars: int = 10_000,
) -> FastAPI:
    if chunk_bytes < 2 or chunk_bytes % 2:
        raise ValueError("chunk_bytes must be a positive even number")

    app = FastAPI(title="Sarashina Japanese TTS WebSocket", docs_url=None, redoc_url=None)
    app.state.backend = backend
    app.state.inference_limiter = asyncio.Semaphore(1)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        ready = bool(api_key and backend is not None and backend.ready)
        reason = None
        if not api_key:
            reason = "WSS_API_KEY is not configured"
        elif backend is None:
            reason = "Sarashina model is not loaded"
        elif not backend.ready:
            reason = backend.not_ready_reason
        return {
            "status": "ok" if ready else "not_ready",
            "service": "sarashina",
            "protocols": ["sarashina"],
            **({"reason": reason} if reason else {}),
        }

    @app.websocket("/tts/sarashina")
    async def sarashina_tts(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization", "")
        if not api_key or not hmac.compare_digest(authorization, f"Bearer {api_key}"):
            await websocket.close(code=4401, reason="valid bearer token required")
            return
        await websocket.accept()

        try:
            while True:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict) or payload.get("type") != "response.create":
                    await websocket.send_json({"type": "error", "message": "expected response.create"})
                    continue
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    await websocket.send_json({"type": "error", "message": "text must be a non-empty string"})
                    continue
                if len(text) > max_text_chars:
                    await websocket.send_json({"type": "error", "message": f"text exceeds {max_text_chars} characters"})
                    continue
                language = payload.get("language")
                if language and str(language).lower() not in {"ja", "jpn", "japanese"}:
                    await websocket.send_json({"type": "error", "message": "Sarashina service only accepts Japanese text"})
                    continue
                if backend is None or not backend.ready:
                    await websocket.send_json({"type": "error", "message": "Sarashina model is not ready"})
                    continue

                try:
                    await websocket.send_json(
                        {
                            "type": "response.audio.started",
                            "format": "pcm_s16le",
                            "sample_rate": backend.sample_rate,
                            "channels": 1,
                        }
                    )
                    async with app.state.inference_limiter:
                        audio = await backend.synthesize(text.strip())
                    if not audio or len(audio) % 2:
                        raise RuntimeError("Sarashina returned invalid PCM16 audio")
                    for offset in range(0, len(audio), chunk_bytes):
                        await websocket.send_bytes(audio[offset : offset + chunk_bytes])
                    await websocket.send_json({"type": "response.completed"})
                except WebSocketDisconnect:
                    return
                except Exception:
                    logger.exception("Sarashina synthesis failed")
                    await websocket.send_json({"type": "error", "message": "Sarashina inference failed"})
        except WebSocketDisconnect:
            return

    return app


def create_app_from_env() -> FastAPI:
    backend = _build_backend_from_env()
    return create_app(
        api_key=os.environ.get("WSS_API_KEY", ""),
        backend=backend,
        chunk_bytes=int(os.environ.get("SARASHINA_AUDIO_CHUNK_BYTES", "4096")),
        max_text_chars=int(os.environ.get("SARASHINA_MAX_TEXT_CHARS", "10000")),
    )
