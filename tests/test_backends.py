import asyncio

import pytest

from sts_ws_server import backends
from sts_ws_server.backends import (
    KotobaWhisperBackend,
    KokoroTTSBackend,
    ModelBackendError,
    OpenAICompatibleTTSBackend,
)
from sts_ws_server.config import ASRConfig, LLMConfig, Settings, TTSConfig


def test_openai_compatible_tts_streams_pcm_and_sends_model_parameters(monkeypatch) -> None:
    captured: dict[str, object] = {}
    captured["clients"] = 0
    captured["payloads"] = []

    class Response:
        is_error = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def aiter_bytes(self, chunk_size):
            captured["chunk_size"] = chunk_size
            yield b"\x01\x00"
            yield b"\x02\x00"

    class Client:
        def __init__(self, **kwargs):
            captured["clients"] += 1
            captured["timeout"] = kwargs["timeout"].read
            self.is_closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def aclose(self):
            captured["closed"] = True
            self.is_closed = True

        def stream(self, method, url, *, headers, json):
            captured.update(method=method, url=url, headers=headers)
            captured["payloads"].append(json)
            return Response()

    monkeypatch.setattr(backends.httpx, "AsyncClient", Client)
    backend = OpenAICompatibleTTSBackend(
        base_url="http://tts.internal/v1/",
        api_key="model-token",
        model="test-voice-model",
        sample_rate=24_000,
        chunk_bytes=2048,
        timeout=30,
    )

    async def collect() -> list[bytes]:
        first = [chunk async for chunk in backend.stream_audio("こんにちは", "voice-a", "ja")]
        second = [chunk async for chunk in backend.stream_audio("second", "voice-a", "ja")]
        await backend.aclose()
        return first + second

    assert asyncio.run(collect()) == [b"\x01\x00", b"\x02\x00"] * 2
    assert captured["clients"] == 1
    assert captured["closed"] is True
    expected = {
        "timeout": 30,
        "method": "POST",
        "url": "http://tts.internal/v1/audio/speech",
        "headers": {"Authorization": "Bearer model-token"},
        "chunk_size": 2048,
    }
    assert {key: captured[key] for key in expected} == expected
    assert captured["payloads"][0] == {
        "model": "test-voice-model",
        "input": "こんにちは",
        "voice": "voice-a",
        "response_format": "pcm",
        "stream_format": "audio",
        "language": "ja",
    }
    assert captured["payloads"][1]["input"] == "second"


def test_kotoba_backend_loads_local_pipeline_and_forces_japanese_transcription(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Pipeline:
        def __call__(
            self,
            audio,
            *,
            chunk_length_s,
            stride_length_s,
            generate_kwargs,
            return_timestamps,
        ):
            captured["audio"] = audio
            captured["chunk_length_s"] = chunk_length_s
            captured["stride_length_s"] = stride_length_s
            captured["generate_kwargs"] = generate_kwargs
            captured["return_timestamps"] = return_timestamps
            return {"text": " こんにちは。 ", "chunks": [{"timestamp": (0.0, 1.0), "text": "こんにちは。"}]}

    class FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()
        inference_mode = staticmethod(lambda: __import__("contextlib").nullcontext())
        from_numpy = staticmethod(lambda audio: audio)

        @staticmethod
        def device(value):
            return value

    class Model:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

    class FakeTransformers:
        class UnusedPipeline:
            def __call__(self, *_args, **_kwargs):
                raise AssertionError("inference should not run for an unsupported language")

        class AutoModelForSpeechSeq2Seq:
            @staticmethod
            def from_pretrained(path, **kwargs):
                captured["model_load"] = (path, kwargs)
                captured["model"] = Model()
                return captured["model"]

        class AutoProcessor:
            @staticmethod
            def from_pretrained(path, **kwargs):
                captured["processor_load"] = (path, kwargs)
                return type("Processor", (), {"tokenizer": "tokenizer", "feature_extractor": "extractor"})()

        @staticmethod
        def pipeline(task, **kwargs):
            captured["pipeline"] = (task, kwargs)
            return Pipeline()

    monkeypatch.setattr(
        "sts_ws_server.backends._load_kotoba_dependencies",
        lambda: (FakeTorch, FakeTransformers.AutoModelForSpeechSeq2Seq, FakeTransformers.AutoProcessor, FakeTransformers.pipeline),
    )
    backend = KotobaWhisperBackend("/models/kotoba", device="cuda:0")
    result = asyncio.run(backend.transcribe(b"\x00\x40" * 16_000, language="ja"))

    assert result.text == "こんにちは。"
    assert result.language == "ja"
    assert result.segments == [{"start": 0.0, "end": 1.0, "text": "こんにちは。"}]
    task, kwargs = captured["pipeline"]
    assert task == "automatic-speech-recognition"
    assert captured["model_load"] == (
        "/models/kotoba",
        {"torch_dtype": "bfloat16", "local_files_only": True, "attn_implementation": "sdpa"},
    )
    assert captured["processor_load"] == ("/models/kotoba", {"local_files_only": True})
    assert captured["device"] == "cuda:0"
    assert captured["eval"] is True
    assert kwargs == {
        "model": captured["model"],
        "tokenizer": "tokenizer",
        "feature_extractor": "extractor",
        "device": "cuda:0",
    }
    assert captured["audio"]["sampling_rate"] == 16_000
    assert captured["generate_kwargs"] == {"language": "ja", "task": "transcribe"}
    assert captured["chunk_length_s"] == 15
    assert captured["stride_length_s"] == (5, 5)
    assert captured["return_timestamps"] is True


def test_kotoba_backend_rejects_non_japanese_language(monkeypatch) -> None:
    class FakeTorch:
        bfloat16 = "bfloat16"
        float32 = "float32"
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()

    class FakeTransformers:
        class UnusedPipeline:
            def __call__(self, *_args, **_kwargs):
                raise AssertionError("inference should not run for an unsupported language")

        class Model:
            def to(self, _device):
                return self

            def eval(self):
                return self

        AutoModelForSpeechSeq2Seq = type(
            "AutoModelForSpeechSeq2Seq",
            (),
            {"from_pretrained": staticmethod(lambda *_args, **_kwargs: FakeTransformers.Model())},
        )
        AutoProcessor = type(
            "AutoProcessor",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda *_args, **_kwargs: type(
                        "Processor", (), {"tokenizer": None, "feature_extractor": None}
                    )()
                )
            },
        )

        @staticmethod
        def pipeline(*_args, **_kwargs):
            return FakeTransformers.UnusedPipeline()

    monkeypatch.setattr(
        "sts_ws_server.backends._load_kotoba_dependencies",
        lambda: (FakeTorch, FakeTransformers.AutoModelForSpeechSeq2Seq, FakeTransformers.AutoProcessor, FakeTransformers.pipeline),
    )
    backend = KotobaWhisperBackend("/models/kotoba", device="cpu")

    with pytest.raises(ModelBackendError, match="Japanese"):
        asyncio.run(backend.transcribe(b"\x00\x00", language="en"))


def test_kotoba_backend_is_selected_from_service_settings(monkeypatch) -> None:
    captured = {}

    class FakeKotoba:
        def __init__(self, model_path, *, device):
            captured.update(model_path=model_path, device=device)

    monkeypatch.setattr(backends, "KotobaWhisperBackend", FakeKotoba)
    backend = backends.create_asr_backend(
        Settings(asr=ASRConfig(backend="kotoba", model="/models/kotoba", device="cuda:0"))
    )

    assert isinstance(backend, FakeKotoba)
    assert captured == {"model_path": "/models/kotoba", "device": "cuda:0"}


def test_settings_read_local_kotoba_and_silero_model_paths(monkeypatch) -> None:
    for name, value in {
        "VAD_BACKEND": "silero",
        "VAD_MODEL_PATH": "/app/sts_ws_server/models/vad/silero/silero_vad.onnx",
        "VAD_THRESHOLD": "0.55",
        "VAD_FRAME_MS": "32",
        "VAD_PADDING_MS": "96",
        "ASR_BACKEND": "kotoba",
        "ASR_MODEL": "/app/sts_ws_server/models/asr/kotoba",
        "ASR_DEVICE": "cuda:0",
        "QWEN_TTS_BACKEND": "openai",
        "QWEN_TTS_BASE_URL": "http://172.19.0.1:8091/v1",
        "QWEN_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "QWEN_TTS_VOICE": "aiden",
        "KOKORO_TTS_BACKEND": "kokoro",
        "KOKORO_TTS_MODEL_PATH": "/app/sts_ws_server/models/tts/kokoro",
        "KOKORO_TTS_DEVICE": "cuda",
        "KOKORO_TTS_VOICE": "jf_alpha",
        "LLM_BACKEND": "openai",
        "LLM_BASE_URL": "http://127.0.0.1:18080/v1",
        "LLM_MODEL": "Ornith-1.5-35B-A3B",
    }.items():
        monkeypatch.setenv(name, value)

    settings = backends.Settings.from_env()

    assert settings.vad_backend == "silero"
    assert settings.vad_model_path == "/app/sts_ws_server/models/vad/silero/silero_vad.onnx"
    assert settings.vad_threshold == 0.55
    assert settings.vad_frame_ms == 32
    assert settings.vad_padding_ms == 96
    assert settings.asr_backend == "kotoba"
    assert settings.asr_model == "/app/sts_ws_server/models/asr/kotoba"
    assert settings.asr_device == "cuda:0"
    assert settings.qwen_tts_backend == "openai"
    assert settings.qwen_tts_base_url == "http://172.19.0.1:8091/v1"
    assert settings.qwen_tts_model == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert settings.qwen_tts_voice == "aiden"
    assert settings.kokoro_tts_backend == "kokoro"
    assert settings.kokoro_tts_model_path == "/app/sts_ws_server/models/tts/kokoro"
    assert settings.kokoro_tts_device == "cuda"
    assert settings.kokoro_tts_voice == "jf_alpha"
    assert settings.llm.backend == "openai"
    assert settings.llm.base_url == "http://127.0.0.1:18080/v1"
    assert settings.llm.model == "Ornith-1.5-35B-A3B"


def test_kokoro_backend_loads_local_model_and_streams_pcm(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "kokoro"
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "kokoro-v1_0.pth").write_bytes(b"model")
    (model_dir / "voices" / "jf_alpha.pt").write_bytes(b"voice")
    captured: dict[str, object] = {}

    class Audio:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.array([0.25, -0.25], dtype=np.float32)

    class Pipeline:
        def __call__(self, text, *, voice):
            captured["request"] = (text, voice)
            return iter([type("Result", (), {"audio": Audio()})()])

    class Model:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    class FakeKModel:
        def __init__(self, **kwargs):
            captured["model"] = kwargs

        def to(self, device):
            captured["device"] = device
            return Model()

    class FakeKPipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

        def __call__(self, text, *, voice):
            return Pipeline()(text, voice=voice)

    monkeypatch.setattr(
        "sts_ws_server.backends._load_kokoro_dependencies",
        lambda: (FakeKModel, FakeKPipeline, __import__("numpy")),
    )
    backend = KokoroTTSBackend(str(model_dir), device="cuda", voice="jf_alpha")

    async def collect() -> list[bytes]:
        return [chunk async for chunk in backend.stream_audio("こんにちは", None, "ja")]

    assert asyncio.run(collect()) == [b"\xff\x1f\x01\xe0"]
    assert captured["device"] == "cuda"
    assert captured["request"] == ("こんにちは", str(model_dir / "voices" / "jf_alpha.pt"))

    async def reject_other_language() -> None:
        with pytest.raises(ModelBackendError, match="Japanese"):
            async for _ in backend.stream_audio("hello", None, "en"):
                pass

    async def reject_file_path() -> None:
        with pytest.raises(ModelBackendError, match="voice"):
            async for _ in backend.stream_audio("こんにちは", "/tmp/other.pt", "ja"):
                pass

    asyncio.run(reject_other_language())
    asyncio.run(reject_file_path())


def test_openai_compatible_llm_backend_streams_content_deltas(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        is_error = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":" world"}}]}'
            yield "data: [DONE]"

    class Client:
        def __init__(self, **kwargs):
            captured["client_timeout"] = kwargs["timeout"].read
            self.is_closed = False

        async def aclose(self):
            captured["closed"] = True
            self.is_closed = True

        def stream(self, method, url, *, headers, json, timeout):
            captured.update(
                method=method,
                url=url,
                headers=headers,
                payload=json,
                connect_timeout=timeout.connect,
                read_timeout=timeout.read,
            )
            return Response()

    monkeypatch.setattr(backends.httpx, "AsyncClient", Client)
    backend = backends.create_llm_backend(
        Settings(
            llm=LLMConfig(
                base_url="http://127.0.0.1:18080/v1/",
                api_key="upstream-secret",
                model="Ornith-1.5-35B-A3B",
                connect_timeout_seconds=3,
                read_timeout_seconds=90,
            )
        )
    )

    async def collect() -> list[str]:
        chunks = [
            chunk
            async for chunk in backend.chat(
                [{"role": "user", "content": "hello"}], temperature=0.2
            )
        ]
        await backend.aclose()
        return chunks

    assert asyncio.run(collect()) == ["Hello", " world"]
    assert captured == {
        "client_timeout": 90,
        "method": "POST",
        "url": "http://127.0.0.1:18080/v1/chat/completions",
        "headers": {"Authorization": "Bearer upstream-secret"},
        "payload": {
            "model": "Ornith-1.5-35B-A3B",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "temperature": 0.2,
        },
        "connect_timeout": 3,
        "read_timeout": 90,
        "closed": True,
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"asr": ASRConfig(max_audio_seconds=0)}, "audio and text limits must be positive"),
        ({"qwen_tts": TTSConfig(max_text_chars=0)}, "audio and text limits must be positive"),
        ({"qwen_tts": TTSConfig(audio_chunk_bytes=1)}, "TTS_AUDIO_CHUNK_BYTES must be an even number of bytes"),
        ({"qwen_tts": TTSConfig(audio_chunk_bytes=3)}, "TTS_AUDIO_CHUNK_BYTES must be an even number of bytes"),
        ({"qwen_tts": TTSConfig(sample_rate=7_999)}, "TTS sample rates must be at least 8000 Hz"),
        ({"cosyvoice_tts": TTSConfig(sample_rate=7_999)}, "TTS sample rates must be at least 8000 Hz"),
    ],
)
def test_settings_reject_invalid_nested_model_limits(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        Settings(**config)
