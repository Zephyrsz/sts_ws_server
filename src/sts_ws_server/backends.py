from __future__ import annotations

import asyncio
import io
import logging
import math
import struct
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from sts_ws_server.config import Settings

logger = logging.getLogger(__name__)


class ModelBackendError(RuntimeError):
    """A safe-to-return inference backend error."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    segments: list[dict[str, object]] | None = None


class ASRBackend(Protocol):
    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript: ...


class TTSBackend(Protocol):
    async def stream_audio(
        self, text: str, voice: str | None, language: str | None
    ) -> AsyncIterator[bytes]: ...


class _PooledHTTPBackend:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


class MockASRBackend:
    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        return Transcript(text="mock transcription", language=language or "und")


class FasterWhisperBackend:
    def __init__(self, settings: Settings) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("ASR_BACKEND=faster-whisper requires the 'asr' extra") from exc

        logger.info(
            "Loading Faster-Whisper model=%s device=%s compute_type=%s",
            settings.asr_model,
            settings.asr_device,
            settings.asr_compute_type,
        )
        self.model = WhisperModel(
            settings.asr_model,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )
        self._lock = asyncio.Lock()

    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe, pcm_s16le, language)

    def _transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        import numpy as np

        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            samples,
            language=language if language and language.lower() != "auto" else None,
            vad_filter=False,
            beam_size=5,
        )
        segment_items = list(segments)
        return Transcript(
            text=" ".join(item.text.strip() for item in segment_items).strip(),
            language=getattr(info, "language", None),
            segments=[
                {"start": float(item.start), "end": float(item.end), "text": item.text.strip()}
                for item in segment_items
            ],
        )


def _load_kotoba_dependencies():
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise RuntimeError("ASR_BACKEND=kotoba requires the 'kotoba' extra") from exc
    return torch, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class KotobaWhisperBackend:
    """Japanese Kotoba Whisper inference using a local Transformers checkpoint."""

    def __init__(self, model_path: str, *, device: str = "cuda:0") -> None:
        torch, auto_model, auto_processor, pipeline_factory = _load_kotoba_dependencies()
        self.device = device
        self.torch = torch
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model_kwargs = {"attn_implementation": "sdpa"} if device.startswith("cuda") else {}
        logger.info("Loading Kotoba Whisper model=%s device=%s", model_path, device)
        model = auto_model.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
            **model_kwargs,
        ).to(device).eval()
        processor = auto_processor.from_pretrained(model_path, local_files_only=True)
        self.pipeline = pipeline_factory(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )
        self._lock = asyncio.Lock()

    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        if language and language.lower() not in {"ja", "jpn", "japanese", "auto"}:
            raise ModelBackendError("Kotoba Whisper only supports Japanese transcription")
        async with self._lock:
            return await asyncio.to_thread(self._transcribe, pcm_s16le)

    def _transcribe(self, pcm_s16le: bytes) -> Transcript:
        import numpy as np

        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        with self.torch.inference_mode():
            result = self.pipeline(
                {"array": samples, "sampling_rate": 16_000},
                chunk_length_s=15,
                stride_length_s=(5, 5),
                generate_kwargs={"language": "ja", "task": "transcribe"},
                return_timestamps=True,
            )
        chunks = result.get("chunks") if isinstance(result, dict) else None
        segments = []
        if isinstance(chunks, list):
            for chunk in chunks:
                timestamp = chunk.get("timestamp")
                text = chunk.get("text")
                if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2 or not isinstance(text, str):
                    continue
                start, end = timestamp
                if start is None or end is None:
                    continue
                segments.append({"start": float(start), "end": float(end), "text": text.strip()})
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str):
            raise ModelBackendError("Kotoba Whisper returned an invalid response")
        return Transcript(text=text.strip(), language="ja", segments=segments)


class OpenAICompatibleASRBackend(_PooledHTTPBackend):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.model_timeout_seconds)
        self.endpoint = f"{settings.asr_base_url.rstrip('/')}/audio/transcriptions"
        self.api_key = settings.asr_api_key
        self.model = settings.asr_http_model

    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(pcm_s16le)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        fields: dict[str, str] = {"model": self.model, "response_format": "verbose_json"}
        if language and language.lower() != "auto":
            fields["language"] = language
        try:
            response = await self._get_client().post(
                self.endpoint,
                headers=headers,
                data=fields,
                files={"file": ("speech.wav", wav_buffer.getvalue(), "audio/wav")},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ModelBackendError(f"ASR backend returned HTTP {exc.response.status_code}") from exc
        except httpx.TimeoutException as exc:
            raise ModelBackendError("ASR backend request timed out") from exc
        except httpx.HTTPError as exc:
            raise ModelBackendError(f"ASR backend transport failed: {type(exc).__name__}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelBackendError("ASR backend returned invalid JSON") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise ModelBackendError("ASR backend response is missing a text field")
        segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
        return Transcript(text=text.strip(), language=payload.get("language"), segments=segments)


class MockTTSBackend:
    def __init__(self, sample_rate: int, chunk_bytes: int) -> None:
        self.sample_rate = sample_rate
        self.chunk_bytes = chunk_bytes

    async def stream_audio(
        self, text: str, voice: str | None, language: str | None
    ) -> AsyncIterator[bytes]:
        duration = min(0.4 + len(text) * 0.025, 2.0)
        sample_count = int(self.sample_rate * duration)
        samples_per_chunk = self.chunk_bytes // 2
        for offset in range(0, sample_count, samples_per_chunk):
            count = min(samples_per_chunk, sample_count - offset)
            samples = [int(1800 * math.sin(2 * math.pi * 440 * (offset + i) / self.sample_rate)) for i in range(count)]
            yield struct.pack("<" + "h" * count, *samples)
            await asyncio.sleep(0)


class OpenAICompatibleTTSBackend(_PooledHTTPBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        sample_rate: int,
        chunk_bytes: int,
        timeout: float,
    ) -> None:
        super().__init__(timeout)
        self.endpoint = f"{base_url.rstrip('/')}/audio/speech"
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_bytes = chunk_bytes

    async def stream_audio(
        self, text: str, voice: str | None, language: str | None
    ) -> AsyncIterator[bytes]:
        payload: dict[str, object] = {
            "model": self.model,
            "input": text,
            "voice": voice or "default",
            "response_format": "pcm",
            "stream_format": "audio",
        }
        if language:
            payload["language"] = language
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with self._get_client().stream("POST", self.endpoint, headers=headers, json=payload) as response:
                if response.is_error:
                    raise ModelBackendError(f"TTS backend returned HTTP {response.status_code}")
                async for chunk in response.aiter_bytes(self.chunk_bytes):
                    if chunk:
                        yield chunk
        except ModelBackendError:
            raise
        except httpx.TimeoutException as exc:
            raise ModelBackendError("TTS backend request timed out") from exc
        except httpx.HTTPError as exc:
            raise ModelBackendError(f"TTS backend transport failed: {type(exc).__name__}") from exc


def create_asr_backend(settings: Settings) -> ASRBackend:
    if settings.asr_backend == "mock":
        return MockASRBackend()
    if settings.asr_backend == "faster-whisper":
        return FasterWhisperBackend(settings)
    if settings.asr_backend == "kotoba":
        return KotobaWhisperBackend(settings.asr_model, device=settings.asr_device)
    if settings.asr_backend == "openai":
        return OpenAICompatibleASRBackend(settings)
    raise ValueError("ASR_BACKEND must be mock, faster-whisper, kotoba, or openai")


def create_tts_backend(settings: Settings, service_name: str) -> TTSBackend:
    if service_name == "qwen-tts":
        backend = settings.qwen_tts_backend
        base_url = settings.qwen_tts_base_url
        api_key = settings.qwen_tts_api_key
        model = settings.qwen_tts_model
        sample_rate = settings.qwen_tts_sample_rate
    elif service_name == "cosyvoice":
        backend = settings.cosyvoice_tts_backend
        base_url = settings.cosyvoice_tts_base_url
        api_key = settings.cosyvoice_tts_api_key
        model = settings.cosyvoice_tts_model
        sample_rate = settings.cosyvoice_tts_sample_rate
    else:
        raise ValueError(f"not a TTS service: {service_name}")

    if backend == "mock":
        return MockTTSBackend(sample_rate, settings.tts_audio_chunk_bytes)
    if backend == "openai":
        return OpenAICompatibleTTSBackend(
            base_url=base_url,
            api_key=api_key,
            model=model,
            sample_rate=sample_rate,
            chunk_bytes=settings.tts_audio_chunk_bytes,
            timeout=settings.model_timeout_seconds,
        )
    raise ValueError(f"{service_name.upper().replace('-', '_')}_TTS_BACKEND must be mock or openai")
