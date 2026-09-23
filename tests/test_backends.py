import asyncio

import pytest

from sts_ws_server import backends
from sts_ws_server.backends import KotobaWhisperBackend, ModelBackendError, OpenAICompatibleTTSBackend


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
        backends.Settings(asr_backend="kotoba", asr_model="/models/kotoba", asr_device="cuda:0")
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
