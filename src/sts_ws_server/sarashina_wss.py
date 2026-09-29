from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from sts_ws_server.logging import ServiceEventLogger

logger = logging.getLogger(__name__)


class SarashinaBackendProtocol(Protocol):
    sample_rate: int
    ready: bool
    not_ready_reason: str

    def resolve_voice(self, voice: str | None) -> str | None: ...

    async def synthesize(self, text: str, voice: str | None = None) -> bytes: ...


class UnknownSarashinaVoice(ValueError):
    pass


@dataclass(frozen=True)
class SarashinaVoiceProfile:
    audio_path: str
    text: str
    label: str = ""


@dataclass(frozen=True)
class SarashinaVoiceConditioning:
    profile: SarashinaVoiceProfile
    flow_embedding: object
    audio_prompt_tokens: object
    audio_prompt_feat: object


def load_voice_profiles(
    config_path: str | Path,
) -> tuple[str, dict[str, SarashinaVoiceProfile]]:
    path = Path(config_path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Sarashina voice profile config must be a JSON object")

    default_voice = data.get("default_voice")
    voice_data = data.get("voices")
    if not isinstance(default_voice, str) or not default_voice:
        raise ValueError("Sarashina voice profile config needs a default_voice")
    if not isinstance(voice_data, dict) or not voice_data:
        raise ValueError("Sarashina voice profile config needs a non-empty voices object")

    profiles: dict[str, SarashinaVoiceProfile] = {}
    for voice_id, values in voice_data.items():
        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ValueError("Sarashina voice IDs must be non-empty strings")
        if not isinstance(values, dict):
            raise ValueError(f"Sarashina voice profile {voice_id!r} must be an object")
        audio = values.get("audio")
        text = values.get("text")
        label = values.get("label", "")
        if not isinstance(audio, str) or not audio.strip():
            raise ValueError(f"Sarashina voice profile {voice_id!r} needs an audio path")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Sarashina voice profile {voice_id!r} needs reference text")
        if not isinstance(label, str):
            raise ValueError(f"Sarashina voice profile {voice_id!r} label must be a string")
        audio_path = Path(audio).expanduser()
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        profiles[voice_id] = SarashinaVoiceProfile(
            str(audio_path.resolve()), text.strip(), label.strip()
        )

    if default_voice not in profiles:
        raise ValueError("Sarashina default_voice must name a configured voice")
    return default_voice, profiles


class SarashinaBackend:
    sample_rate = 24_000

    def __init__(
        self,
        generator,
        *,
        prompt_path: str = "",
        prompt_text: str = "",
        voice_profiles: Mapping[str, SarashinaVoiceProfile] | None = None,
        default_voice: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        self.generator = generator
        self.prompt_path = prompt_path
        self.prompt_text = prompt_text.strip()
        self.max_tokens = max_tokens
        self._lock = asyncio.Lock()
        self.default_voice = default_voice
        self.voice_profiles = dict(voice_profiles or {})
        self.voice_conditioning: dict[str, SarashinaVoiceConditioning] = {}
        self.ready = True
        self.not_ready_reason = ""
        self.flow_embedding = None
        self.audio_prompt_tokens = None
        self.audio_prompt_feat = None

        if self.voice_profiles:
            if self.default_voice not in self.voice_profiles:
                self.ready = False
                self.not_ready_reason = "default Sarashina voice is not configured"
                return
            self._load_voice_conditioning()
            return

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

    @property
    def available_voices(self) -> tuple[str, ...]:
        return tuple(self.voice_profiles)

    def _load_voice_conditioning(self) -> None:
        for voice_id, profile in self.voice_profiles.items():
            if not profile.audio_path or not profile.text:
                self.ready = False
                self.not_ready_reason = f"Sarashina voice {voice_id!r} needs audio and reference text"
                self.voice_conditioning.clear()
                return
            if not Path(profile.audio_path).is_file():
                self.ready = False
                self.not_ready_reason = f"Sarashina reference audio is not available for voice {voice_id!r}"
                self.voice_conditioning.clear()
                return
        try:
            for voice_id, profile in self.voice_profiles.items():
                self.voice_conditioning[voice_id] = SarashinaVoiceConditioning(
                    profile=profile,
                    flow_embedding=self.generator._extract_zero_shot_embedding(
                        profile.audio_path
                    ),
                    audio_prompt_tokens=self.generator._extract_audio_prompt_tokens(
                        profile.audio_path
                    ),
                    audio_prompt_feat=self.generator._extract_audio_prompt_feat(
                        profile.audio_path
                    ),
                )
        except Exception as exc:
            self.voice_conditioning.clear()
            self.ready = False
            self.not_ready_reason = f"failed to load Sarashina voice profiles: {exc}"
            logger.exception("Failed to load Sarashina voice profile conditioning")

    def resolve_voice(self, voice: str | None) -> str | None:
        selected = voice or self.default_voice
        if selected is not None and selected not in self.voice_profiles:
            raise UnknownSarashinaVoice(f"unknown Sarashina voice: {selected}")
        return selected

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize, text, voice)

    def _synthesize(self, text: str, voice: str | None = None) -> bytes:
        selected = self.resolve_voice(voice)
        conditioning = self.voice_conditioning.get(selected) if selected else None
        if conditioning is None and selected is None:
            prompt_path = self.prompt_path
            prompt_text = self.prompt_text
            flow_embedding = self.flow_embedding
            audio_prompt_tokens = self.audio_prompt_tokens
            audio_prompt_feat = self.audio_prompt_feat
        elif conditioning is None:
            raise UnknownSarashinaVoice(f"unknown Sarashina voice: {selected}")
        else:
            prompt_path = conditioning.profile.audio_path
            prompt_text = conditioning.profile.text
            flow_embedding = conditioning.flow_embedding
            audio_prompt_tokens = conditioning.audio_prompt_tokens
            audio_prompt_feat = conditioning.audio_prompt_feat
        wavs = self.generator.generate(
            texts=[text],
            flow_embedding=flow_embedding,
            audio_prompt_text=prompt_text or None,
            audio_prompt_tokens=audio_prompt_tokens,
            audio_prompt_feat=audio_prompt_feat,
            audio_prompt_path=prompt_path or None,
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
    profiles_path = os.environ.get("SARASHINA_VOICE_PROFILES", "").strip()
    if profiles_path:
        default_voice, profiles = load_voice_profiles(profiles_path)
        return SarashinaBackend(
            generator,
            voice_profiles=profiles,
            default_voice=default_voice,
            max_tokens=int(os.environ.get("SARASHINA_MAX_TOKENS", "2048")),
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

    event_logger = ServiceEventLogger.from_environment("sarashina-ws")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            event_logger.close()

    app = FastAPI(
        title="Sarashina Japanese TTS WebSocket",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
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
        connection_id = uuid4().hex
        connection_fields = {
            "connection_id": connection_id,
            "client_ip": getattr(websocket.client, "host", None) if websocket.client else None,
            "path": websocket.url.path,
        }
        authorization = websocket.headers.get("authorization", "")
        if not api_key or not hmac.compare_digest(authorization, f"Bearer {api_key}"):
            event_logger.event("connection.rejected", reason="unauthorized", authorized=False, **connection_fields)
            await websocket.close(code=4401, reason="valid bearer token required")
            return
        await websocket.accept()
        event_logger.event("connection.accepted", authorized=True, **connection_fields)

        async def send_json(payload: dict[str, object]) -> None:
            await websocket.send_json(payload)
            event_logger.event("output.json", payload=payload, **connection_fields)

        try:
            while True:
                payload = await websocket.receive_json()
                event_logger.event("input.json", payload=payload, **connection_fields)
                if not isinstance(payload, dict) or payload.get("type") != "response.create":
                    await send_json({"type": "error", "message": "expected response.create"})
                    continue
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    await send_json({"type": "error", "message": "text must be a non-empty string"})
                    continue
                if len(text) > max_text_chars:
                    await send_json({"type": "error", "message": f"text exceeds {max_text_chars} characters"})
                    continue
                voice = payload.get("voice")
                if voice is not None and (not isinstance(voice, str) or not voice.strip()):
                    await send_json({"type": "error", "message": "voice must be a non-empty string"})
                    continue
                language = payload.get("language")
                if language and str(language).lower() not in {"ja", "jpn", "japanese"}:
                    await send_json({"type": "error", "message": "Sarashina service only accepts Japanese text"})
                    continue
                if backend is None or not backend.ready:
                    await send_json({"type": "error", "message": "Sarashina model is not ready"})
                    continue
                try:
                    resolver = getattr(backend, "resolve_voice", None)
                    selected_voice = resolver(voice.strip() if voice else None) if resolver else voice
                except UnknownSarashinaVoice as exc:
                    await send_json({"type": "error", "message": str(exc)})
                    continue

                try:
                    await send_json(
                        {
                            "type": "response.audio.started",
                            "format": "pcm_s16le",
                            "sample_rate": backend.sample_rate,
                            "channels": 1,
                        }
                    )
                    async with app.state.inference_limiter:
                        audio = await backend.synthesize(text.strip(), selected_voice)
                    if not audio or len(audio) % 2:
                        raise RuntimeError("Sarashina returned invalid PCM16 audio")
                    event_logger.event(
                        "output.audio",
                        format="pcm_s16le",
                        sample_rate=backend.sample_rate,
                        bytes=len(audio),
                        frames=len(audio) // 2,
                        chunks=(len(audio) + chunk_bytes - 1) // chunk_bytes,
                        **connection_fields,
                    )
                    for offset in range(0, len(audio), chunk_bytes):
                        await websocket.send_bytes(audio[offset : offset + chunk_bytes])
                    await send_json({"type": "response.completed"})
                except WebSocketDisconnect:
                    return
                except Exception:
                    logger.exception("Sarashina synthesis failed")
                    await send_json({"type": "error", "message": "Sarashina inference failed"})
        except WebSocketDisconnect:
            return
        finally:
            event_logger.event("connection.closed", **connection_fields)

    return app


def create_app_from_env() -> FastAPI:
    backend = _build_backend_from_env()
    return create_app(
        api_key=os.environ.get("WSS_API_KEY", ""),
        backend=backend,
        chunk_bytes=int(os.environ.get("SARASHINA_AUDIO_CHUNK_BYTES", "4096")),
        max_text_chars=int(os.environ.get("SARASHINA_MAX_TEXT_CHARS", "10000")),
    )
