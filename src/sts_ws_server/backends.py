from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import struct
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
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


def _load_qwen3_asr_dependencies():
    try:
        import numpy as np
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "ASR_BACKEND=qwen3 requires the isolated qwen-asr runtime"
        ) from exc
    return np, torch, Qwen3ASRModel


class Qwen3ASRBackend:
    """Local Qwen3-ASR inference through the official qwen-asr wrapper."""

    def __init__(self, settings: Settings) -> None:
        np, torch, model_factory = _load_qwen3_asr_dependencies()
        self.np = np
        self.torch = torch
        self.device = self._resolve_device(torch, settings.asr_device)
        self.dtype = self._resolve_dtype(torch, settings.asr_compute_type, self.device)
        self.max_new_tokens = settings.asr_max_new_tokens
        model_path = settings.asr_model
        logger.info(
            "Loading Qwen3-ASR model=%s device=%s dtype=%s max_new_tokens=%s",
            model_path,
            self.device,
            self.dtype,
            self.max_new_tokens,
        )
        self.model = model_factory.from_pretrained(
            model_path,
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=1,
            max_new_tokens=self.max_new_tokens,
        )
        self._lock = asyncio.Lock()

    @staticmethod
    def _resolve_device(torch, configured: str) -> str:
        if configured != "auto":
            return configured
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(torch, configured: str, device: str):
        if configured in {"float16", "float32", "bfloat16"}:
            return getattr(torch, configured)
        if device.startswith("cuda"):
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _language_code(detected: object) -> str | None:
        if not isinstance(detected, str) or not detected.strip():
            return None
        normalized = detected.strip().lower()
        language_codes = {
            "chinese": "zh",
            "english": "en",
            "cantonese": "yue",
            "arabic": "ar",
            "german": "de",
            "french": "fr",
            "spanish": "es",
            "portuguese": "pt",
            "indonesian": "id",
            "italian": "it",
            "korean": "ko",
            "russian": "ru",
            "thai": "th",
            "vietnamese": "vi",
            "japanese": "ja",
            "turkish": "tr",
            "hindi": "hi",
            "malay": "ms",
            "dutch": "nl",
            "swedish": "sv",
            "danish": "da",
            "finnish": "fi",
            "polish": "pl",
            "czech": "cs",
            "filipino": "fil",
            "persian": "fa",
            "greek": "el",
            "romanian": "ro",
            "hungarian": "hu",
            "macedonian": "mk",
        }
        return normalized if len(normalized) in {2, 3} else language_codes.get(normalized, detected.strip())

    def _transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        samples = self.np.frombuffer(pcm_s16le, dtype="<i2").astype(self.np.float32) / 32768.0
        forced_language = language if language and language.lower() != "auto" else None
        language_name = self._language_name(forced_language)
        result = self.model.transcribe(
            audio=(samples, 16_000),
            language=language_name,
            return_time_stamps=False,
        )[0]
        return Transcript(
            text=result.text.strip(),
            language=self._language_code(result.language),
            segments=[],
        )

    @staticmethod
    def _language_name(language: str | None) -> str | None:
        if not language or language.lower() == "auto":
            return None
        names = {
            "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
            "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
            "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
            "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
            "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
            "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
            "fil": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian",
            "hu": "Hungarian", "mk": "Macedonian",
        }
        return names.get(language.lower(), language)

    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe, pcm_s16le, language)


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


def _load_kokoro_dependencies():
    try:
        import numpy as np
        from kokoro import KModel, KPipeline
    except ImportError as exc:
        raise RuntimeError(
            "KOKORO_TTS_BACKEND=kokoro requires the 'kokoro' extra"
        ) from exc
    return KModel, KPipeline, np


class KokoroTTSBackend:
    """Local Japanese Kokoro-82M inference with PCM16 streaming output."""

    sample_rate = 24_000

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        voice: str = "jf_alpha",
        chunk_bytes: int = 4096,
    ) -> None:
        if not model_path:
            raise RuntimeError("KOKORO_TTS_MODEL_PATH is required")
        root = Path(model_path)
        model_file = root / "kokoro-v1_0.pth"
        config_file = root / "config.json"
        if not model_file.is_file() or not config_file.is_file():
            raise RuntimeError(
                f"Kokoro model directory must contain {model_file.name} and {config_file.name}"
            )
        KModel, KPipeline, self.np = _load_kokoro_dependencies()
        logger.info("Loading Kokoro model=%s device=%s", model_path, device)
        model = KModel(config=str(config_file), model=str(model_file)).to(device).eval()
        self.pipeline = KPipeline(
            lang_code="j", repo_id=str(root), model=model, device=device
        )
        self.model_root = root
        self.device = device
        self.default_voice = voice
        self.chunk_bytes = chunk_bytes
        self._lock = asyncio.Lock()

    def _voice_path(self, voice: str | None) -> str:
        selected = voice or self.default_voice
        if not selected.startswith(("jf_", "jm_")) or not selected.replace("_", "").isalnum():
            raise ModelBackendError("Kokoro voice must be a Japanese voice name")
        voice_path = self.model_root / "voices" / f"{selected}.pt"
        if not voice_path.is_file():
            raise ModelBackendError(f"Kokoro voice is unavailable: {selected}")
        return str(voice_path)

    def _pcm_segments(self, text: str, voice: str | None):
        voice_path = self._voice_path(voice)
        for result in self.pipeline(text, voice=voice_path):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            samples = audio.detach().float().cpu().numpy()
            samples = self.np.clip(samples, -1.0, 1.0)
            yield (samples * 32767.0).astype("<i2").tobytes()

    async def stream_audio(
        self, text: str, voice: str | None, language: str | None
    ) -> AsyncIterator[bytes]:
        if language and language.lower() not in {"ja", "jpn", "japanese"}:
            raise ModelBackendError("Kokoro Japanese TTS only supports Japanese")
        if not text.strip():
            raise ModelBackendError("Kokoro text must not be empty")
        async with self._lock:
            segments = self._pcm_segments(text, voice)
            sent_audio = False
            while True:
                segment = await asyncio.to_thread(next, segments, None)
                if segment is None:
                    break
                for offset in range(0, len(segment), self.chunk_bytes):
                    sent_audio = True
                    yield segment[offset : offset + self.chunk_bytes]
            if not sent_audio:
                raise ModelBackendError("Kokoro returned no audio")


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


class LLMBackend(Protocol):
    """统一的 LLM Backend 协议"""
    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = True,
        **kwargs
    ) -> AsyncIterator[str]: ...


class OpenAICompatibleLLMBackend(_PooledHTTPBackend):
    """OpenAI 兼容的 LLM Backend"""

    def __init__(self, settings: Settings) -> None:
        # 使用 LLM 配置的超时
        super().__init__(settings.llm.read_timeout_seconds)
        self.endpoint = f"{settings.llm.base_url.rstrip('/')}/chat/completions"
        self.api_key = settings.llm.api_key
        self.model = settings.llm.model
        self.connect_timeout = settings.llm.connect_timeout_seconds

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = True,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式 LLM 调用"""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            timeout = httpx.Timeout(
                connect=self.connect_timeout,
                read=self.timeout,
                write=10.0,
                pool=5.0
            )

            async with self._get_client().stream(
                "POST",
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=timeout
            ) as response:
                if response.is_error:
                    error_text = await response.aread()
                    raise ModelBackendError(
                        f"LLM backend returned HTTP {response.status_code}: {error_text.decode()}"
                    )

                async for line in response.aiter_lines():
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            chunk = line[6:]  # 去掉 "data: " 前缀
                            data = json.loads(chunk) if isinstance(chunk, str) else chunk

                            if isinstance(data, dict):
                                choices = data.get("choices", [])
                                if choices and isinstance(choices, list):
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except ModelBackendError:
            raise
        except httpx.TimeoutException as exc:
            raise ModelBackendError("LLM backend request timed out") from exc
        except httpx.HTTPError as exc:
            raise ModelBackendError(f"LLM backend transport failed: {type(exc).__name__}") from exc


def create_asr_backend(settings: Settings) -> ASRBackend:
    if settings.asr_backend == "mock":
        return MockASRBackend()
    if settings.asr_backend == "faster-whisper":
        return FasterWhisperBackend(settings)
    if settings.asr_backend in {"qwen3", "qwen3-asr"}:
        return Qwen3ASRBackend(settings)
    if settings.asr_backend == "kotoba":
        return KotobaWhisperBackend(settings.asr_model, device=settings.asr_device)
    if settings.asr_backend == "openai":
        return OpenAICompatibleASRBackend(settings)
    raise ValueError("ASR_BACKEND must be mock, faster-whisper, qwen3, kotoba, or openai")


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
    elif service_name == "kokoro":
        if settings.kokoro_tts_backend == "mock":
            return MockTTSBackend(
                settings.kokoro_tts_sample_rate, settings.kokoro_tts.audio_chunk_bytes
            )
        if settings.kokoro_tts_backend == "kokoro":
            return KokoroTTSBackend(
                settings.kokoro_tts_model_path,
                device=settings.kokoro_tts_device,
                voice=settings.kokoro_tts_voice,
                chunk_bytes=settings.kokoro_tts.audio_chunk_bytes,
            )
        raise ValueError("KOKORO_TTS_BACKEND must be mock or kokoro")
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


def create_llm_backend(settings: Settings) -> LLMBackend:
    """创建 LLM Backend (统一架构)"""
    if settings.llm.backend == "openai":
        return OpenAICompatibleLLMBackend(settings)
    raise ValueError("LLM_BACKEND must be openai")
