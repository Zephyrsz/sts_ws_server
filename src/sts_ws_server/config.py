from __future__ import annotations

import os
from dataclasses import dataclass

SERVICES = {"all", "asr", "qwen-tts", "cosyvoice"}


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return default if raw_value is None else int(raw_value)


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    return default if raw_value is None else float(raw_value)


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    service_name: str = "all"
    max_audio_seconds: int = 120
    max_text_chars: int = 10_000
    max_concurrent_requests: int = 2
    vad_backend: str = "energy"
    vad_model_path: str = ""
    vad_threshold: float = 0.5
    vad_provider: str = "CPUExecutionProvider"
    vad_rms_threshold: float = 450.0
    vad_frame_ms: int = 20
    vad_padding_ms: int = 120
    asr_backend: str = "mock"
    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_base_url: str = "http://host.docker.internal:8090/v1"
    asr_api_key: str = ""
    asr_http_model: str = "whisper-1"
    qwen_tts_backend: str = "mock"
    qwen_tts_base_url: str = "http://host.docker.internal:8091/v1"
    qwen_tts_api_key: str = ""
    qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    qwen_tts_sample_rate: int = 24_000
    qwen_tts_voice: str = "aiden"
    cosyvoice_tts_backend: str = "mock"
    cosyvoice_tts_base_url: str = "http://host.docker.internal:8092/v1"
    cosyvoice_tts_api_key: str = ""
    cosyvoice_tts_model: str = "cosyvoice3"
    cosyvoice_tts_sample_rate: int = 24_000
    cosyvoice_tts_voice: str = "default"
    tts_audio_chunk_bytes: int = 4096
    model_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.service_name not in SERVICES:
            raise ValueError(f"SERVICE_NAME must be one of: {', '.join(sorted(SERVICES))}")
        if self.max_audio_seconds < 1 or self.max_text_chars < 1:
            raise ValueError("audio and text limits must be positive")
        if self.max_concurrent_requests < 1:
            raise ValueError("MAX_CONCURRENT_REQUESTS must be positive")
        if self.tts_audio_chunk_bytes < 2 or self.tts_audio_chunk_bytes % 2:
            raise ValueError("TTS_AUDIO_CHUNK_BYTES must be an even number of bytes")
        if self.qwen_tts_sample_rate < 8_000 or self.cosyvoice_tts_sample_rate < 8_000:
            raise ValueError("TTS sample rates must be at least 8000 Hz")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_key=os.getenv("WSS_API_KEY", ""),
            service_name=os.getenv("SERVICE_NAME", "all"),
            max_audio_seconds=_int_env("MAX_AUDIO_SECONDS", 120),
            max_text_chars=_int_env("MAX_TEXT_CHARS", 10_000),
            max_concurrent_requests=_int_env("MAX_CONCURRENT_REQUESTS", 2),
            vad_backend=os.getenv("VAD_BACKEND", "energy"),
            vad_model_path=os.getenv("VAD_MODEL_PATH", ""),
            vad_threshold=_float_env("VAD_THRESHOLD", 0.5),
            vad_provider=os.getenv("VAD_ONNX_PROVIDER", "CPUExecutionProvider"),
            vad_rms_threshold=_float_env("VAD_RMS_THRESHOLD", 450.0),
            vad_frame_ms=_int_env("VAD_FRAME_MS", 20),
            vad_padding_ms=_int_env("VAD_PADDING_MS", 120),
            asr_backend=os.getenv("ASR_BACKEND", "mock"),
            asr_model=os.getenv("ASR_MODEL", "small"),
            asr_device=os.getenv("ASR_DEVICE", "cpu"),
            asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", "int8"),
            asr_base_url=os.getenv("ASR_BASE_URL", "http://host.docker.internal:8090/v1"),
            asr_api_key=os.getenv("ASR_API_KEY", ""),
            asr_http_model=os.getenv("ASR_HTTP_MODEL", "whisper-1"),
            qwen_tts_backend=os.getenv("QWEN_TTS_BACKEND", "mock"),
            qwen_tts_base_url=os.getenv("QWEN_TTS_BASE_URL", "http://host.docker.internal:8091/v1"),
            qwen_tts_api_key=os.getenv("QWEN_TTS_API_KEY", ""),
            qwen_tts_model=os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            qwen_tts_sample_rate=_int_env("QWEN_TTS_SAMPLE_RATE", 24_000),
            qwen_tts_voice=os.getenv("QWEN_TTS_VOICE", "aiden"),
            cosyvoice_tts_backend=os.getenv("COSYVOICE_TTS_BACKEND", "mock"),
            cosyvoice_tts_base_url=os.getenv(
                "COSYVOICE_TTS_BASE_URL", "http://host.docker.internal:8092/v1"
            ),
            cosyvoice_tts_api_key=os.getenv("COSYVOICE_TTS_API_KEY", ""),
            cosyvoice_tts_model=os.getenv("COSYVOICE_TTS_MODEL", "cosyvoice3"),
            cosyvoice_tts_sample_rate=_int_env("COSYVOICE_TTS_SAMPLE_RATE", 24_000),
            cosyvoice_tts_voice=os.getenv("COSYVOICE_TTS_VOICE", "default"),
            tts_audio_chunk_bytes=_int_env("TTS_AUDIO_CHUNK_BYTES", 4096),
            model_timeout_seconds=_float_env("MODEL_TIMEOUT_SECONDS", 300.0),
        )
