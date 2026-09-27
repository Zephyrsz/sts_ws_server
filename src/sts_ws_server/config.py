from __future__ import annotations

import os
from dataclasses import dataclass, field

# 服务类型
SERVICES = {"all", "asr", "qwen-tts", "cosyvoice", "kokoro", "llm"}


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return default if raw_value is None else int(raw_value)


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    return default if raw_value is None else float(raw_value)


@dataclass(frozen=True)
class CommonConfig:
    """通用配置 - 所有服务共享"""
    # 统一认证
    api_key: str = ""

    # 资源限制
    max_concurrent_requests: int = 2
    model_timeout_seconds: float = 300.0

    # VAD 配置 (仅 ASR 使用)
    vad_backend: str = "energy"
    vad_model_path: str = ""
    vad_threshold: float = 0.5
    vad_provider: str = "CPUExecutionProvider"
    vad_rms_threshold: float = 450.0
    vad_frame_ms: int = 20
    vad_padding_ms: int = 120


@dataclass(frozen=True)
class ASRConfig:
    """ASR 服务配置"""
    backend: str = "mock"
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    max_new_tokens: int = 256

    # HTTP backend 配置
    base_url: str = "http://host.docker.internal:8090/v1"
    api_key: str = ""
    http_model: str = "whisper-1"

    # 限制
    max_audio_seconds: int = 120


@dataclass(frozen=True)
class TTSConfig:
    """TTS 服务配置"""
    backend: str = "mock"
    base_url: str = "http://host.docker.internal:8091/v1"
    api_key: str = ""
    model: str = "tts-model"
    model_path: str = ""
    device: str = "cpu"
    language: str = "ja"
    sample_rate: int = 24_000
    voice: str = "default"

    # 限制
    max_text_chars: int = 10_000
    audio_chunk_bytes: int = 4096


@dataclass(frozen=True)
class LLMConfig:
    """LLM 服务配置"""
    backend: str = "openai"
    base_url: str = "http://127.0.0.1:18080/v1"
    api_key: str = ""
    model: str = "llm-model"

    # 超时配置
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 600.0


@dataclass(frozen=True)
class Settings:
    """统一配置"""
    service_name: str = "all"
    common: CommonConfig = field(default_factory=CommonConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    qwen_tts: TTSConfig = field(default_factory=TTSConfig)
    cosyvoice_tts: TTSConfig = field(default_factory=TTSConfig)
    kokoro_tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self) -> None:
        if self.service_name not in SERVICES:
            raise ValueError(f"SERVICE_NAME must be one of: {', '.join(sorted(SERVICES))}")
        if self.common.max_concurrent_requests < 1:
            raise ValueError("MAX_CONCURRENT_REQUESTS must be positive")
        if (
            self.asr.max_audio_seconds < 1
            or self.qwen_tts.max_text_chars < 1
            or self.cosyvoice_tts.max_text_chars < 1
            or self.kokoro_tts.max_text_chars < 1
        ):
            raise ValueError("audio and text limits must be positive")
        for tts_config in (self.qwen_tts, self.cosyvoice_tts, self.kokoro_tts):
            if tts_config.audio_chunk_bytes < 2 or tts_config.audio_chunk_bytes % 2:
                raise ValueError("TTS_AUDIO_CHUNK_BYTES must be an even number of bytes")
            if tts_config.sample_rate < 8_000:
                raise ValueError("TTS sample rates must be at least 8000 Hz")

    @property
    def api_key(self) -> str:
        """统一的 API Key (向后兼容)"""
        return self.common.api_key

    @property
    def max_concurrent_requests(self) -> int:
        """统一的并发限制 (向后兼容)"""
        return self.common.max_concurrent_requests

    @property
    def model_timeout_seconds(self) -> float:
        """统一的超时配置 (向后兼容)"""
        return self.common.model_timeout_seconds

    # VAD 配置 (向后兼容)
    @property
    def vad_backend(self) -> str:
        return self.common.vad_backend

    @property
    def vad_model_path(self) -> str:
        return self.common.vad_model_path

    @property
    def vad_threshold(self) -> float:
        return self.common.vad_threshold

    @property
    def vad_provider(self) -> str:
        return self.common.vad_provider

    @property
    def vad_rms_threshold(self) -> float:
        return self.common.vad_rms_threshold

    @property
    def vad_frame_ms(self) -> int:
        return self.common.vad_frame_ms

    @property
    def vad_padding_ms(self) -> int:
        return self.common.vad_padding_ms

    # ASR 配置 (向后兼容)
    @property
    def asr_backend(self) -> str:
        return self.asr.backend

    @property
    def asr_model(self) -> str:
        return self.asr.model

    @property
    def asr_device(self) -> str:
        return self.asr.device

    @property
    def asr_compute_type(self) -> str:
        return self.asr.compute_type

    @property
    def asr_max_new_tokens(self) -> int:
        return self.asr.max_new_tokens

    @property
    def asr_base_url(self) -> str:
        return self.asr.base_url

    @property
    def asr_api_key(self) -> str:
        return self.asr.api_key

    @property
    def asr_http_model(self) -> str:
        return self.asr.http_model

    @property
    def max_audio_seconds(self) -> int:
        return self.asr.max_audio_seconds

    # TTS 配置 (向后兼容)
    @property
    def qwen_tts_backend(self) -> str:
        return self.qwen_tts.backend

    @property
    def qwen_tts_base_url(self) -> str:
        return self.qwen_tts.base_url

    @property
    def qwen_tts_api_key(self) -> str:
        return self.qwen_tts.api_key

    @property
    def qwen_tts_model(self) -> str:
        return self.qwen_tts.model

    @property
    def qwen_tts_sample_rate(self) -> int:
        return self.qwen_tts.sample_rate

    @property
    def qwen_tts_voice(self) -> str:
        return self.qwen_tts.voice

    @property
    def cosyvoice_tts_backend(self) -> str:
        return self.cosyvoice_tts.backend

    @property
    def cosyvoice_tts_base_url(self) -> str:
        return self.cosyvoice_tts.base_url

    @property
    def cosyvoice_tts_api_key(self) -> str:
        return self.cosyvoice_tts.api_key

    @property
    def cosyvoice_tts_model(self) -> str:
        return self.cosyvoice_tts.model

    @property
    def cosyvoice_tts_sample_rate(self) -> int:
        return self.cosyvoice_tts.sample_rate

    @property
    def cosyvoice_tts_voice(self) -> str:
        return self.cosyvoice_tts.voice

    @property
    def kokoro_tts_backend(self) -> str:
        return self.kokoro_tts.backend

    @property
    def kokoro_tts_model_path(self) -> str:
        return self.kokoro_tts.model_path

    @property
    def kokoro_tts_device(self) -> str:
        return self.kokoro_tts.device

    @property
    def kokoro_tts_language(self) -> str:
        return self.kokoro_tts.language

    @property
    def kokoro_tts_voice(self) -> str:
        return self.kokoro_tts.voice

    @property
    def kokoro_tts_sample_rate(self) -> int:
        return self.kokoro_tts.sample_rate

    @property
    def max_text_chars(self) -> int:
        return self.qwen_tts.max_text_chars

    @property
    def tts_audio_chunk_bytes(self) -> int:
        return self.qwen_tts.audio_chunk_bytes

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量加载配置"""
        # 统一的 API Key
        unified_api_key = os.getenv("WSS_API_KEY", "")

        return cls(
            service_name=os.getenv("SERVICE_NAME", "all"),

            # 通用配置
            common=CommonConfig(
                api_key=unified_api_key,
                max_concurrent_requests=_int_env("MAX_CONCURRENT_REQUESTS", 2),
                model_timeout_seconds=_float_env("MODEL_TIMEOUT_SECONDS", 300.0),
                vad_backend=os.getenv("VAD_BACKEND", "energy"),
                vad_model_path=os.getenv("VAD_MODEL_PATH", ""),
                vad_threshold=_float_env("VAD_THRESHOLD", 0.5),
                vad_provider=os.getenv("VAD_ONNX_PROVIDER", "CPUExecutionProvider"),
                vad_rms_threshold=_float_env("VAD_RMS_THRESHOLD", 450.0),
                vad_frame_ms=_int_env("VAD_FRAME_MS", 20),
                vad_padding_ms=_int_env("VAD_PADDING_MS", 120),
            ),

            # ASR 配置
            asr=ASRConfig(
                backend=os.getenv("ASR_BACKEND", "mock"),
                model=os.getenv("ASR_MODEL", "small"),
                device=os.getenv("ASR_DEVICE", "cpu"),
                compute_type=os.getenv("ASR_COMPUTE_TYPE", "int8"),
                max_new_tokens=_int_env("ASR_MAX_NEW_TOKENS", 256),
                base_url=os.getenv("ASR_BASE_URL", "http://host.docker.internal:8090/v1"),
                api_key=os.getenv("ASR_API_KEY", ""),
                http_model=os.getenv("ASR_HTTP_MODEL", "whisper-1"),
                max_audio_seconds=_int_env("MAX_AUDIO_SECONDS", 120),
            ),

            # Qwen TTS 配置
            qwen_tts=TTSConfig(
                backend=os.getenv("QWEN_TTS_BACKEND", "mock"),
                base_url=os.getenv("QWEN_TTS_BASE_URL", "http://host.docker.internal:8091/v1"),
                api_key=os.getenv("QWEN_TTS_API_KEY", ""),
                model=os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
                sample_rate=_int_env("QWEN_TTS_SAMPLE_RATE", 24_000),
                voice=os.getenv("QWEN_TTS_VOICE", "aiden"),
                max_text_chars=_int_env("MAX_TEXT_CHARS", 10_000),
                audio_chunk_bytes=_int_env("TTS_AUDIO_CHUNK_BYTES", 4096),
            ),

            # CosyVoice TTS 配置
            cosyvoice_tts=TTSConfig(
                backend=os.getenv("COSYVOICE_TTS_BACKEND", "mock"),
                base_url=os.getenv("COSYVOICE_TTS_BASE_URL", "http://host.docker.internal:8092/v1"),
                api_key=os.getenv("COSYVOICE_TTS_API_KEY", ""),
                model=os.getenv("COSYVOICE_TTS_MODEL", "cosyvoice3"),
                sample_rate=_int_env("COSYVOICE_TTS_SAMPLE_RATE", 24_000),
                voice=os.getenv("COSYVOICE_TTS_VOICE", "default"),
                max_text_chars=_int_env("MAX_TEXT_CHARS", 10_000),
                audio_chunk_bytes=_int_env("TTS_AUDIO_CHUNK_BYTES", 4096),
            ),

            # Kokoro Japanese TTS configuration
            kokoro_tts=TTSConfig(
                backend=os.getenv("KOKORO_TTS_BACKEND", "mock"),
                model_path=os.getenv("KOKORO_TTS_MODEL_PATH", ""),
                device=os.getenv("KOKORO_TTS_DEVICE", "cuda"),
                language=os.getenv("KOKORO_TTS_LANGUAGE", "ja"),
                voice=os.getenv("KOKORO_TTS_VOICE", "jf_alpha"),
                sample_rate=_int_env("KOKORO_TTS_SAMPLE_RATE", 24_000),
                max_text_chars=_int_env("KOKORO_TTS_MAX_TEXT_CHARS", 10_000),
                audio_chunk_bytes=_int_env("KOKORO_TTS_AUDIO_CHUNK_BYTES", 4096),
            ),

            # LLM 配置
            llm=LLMConfig(
                backend=os.getenv("LLM_BACKEND", "openai"),
                base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:18080/v1"),
                api_key=os.getenv("LLM_API_KEY", ""),
                model=os.getenv("LLM_MODEL", "Ornith-1.5-35B-A3B"),
                connect_timeout_seconds=_float_env("LLM_CONNECT_TIMEOUT_SECONDS", 10.0),
                read_timeout_seconds=_float_env("LLM_READ_TIMEOUT_SECONDS", 600.0),
            ),
        )
