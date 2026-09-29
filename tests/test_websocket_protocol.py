import json
import struct

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sts_ws_server import app as app_module
from sts_ws_server.app import create_app
from sts_ws_server.backends import Transcript
from sts_ws_server.config import CommonConfig, Settings


class RecordingASR:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None]] = []

    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript:
        self.calls.append((pcm_s16le, language))
        return Transcript(text="recognized speech", language=language or "ja")


class ChunkTTS:
    sample_rate = 24_000

    async def stream_audio(self, text: str, voice: str | None, language: str | None):
        yield b"\x01\x00" * 256
        yield b"\x02\x00" * 256


def make_audio(amplitude: int, frames: int = 8) -> bytes:
    frame = struct.pack("<" + "h" * 320, *([amplitude] * 320))
    return frame * frames


def make_client(service_name: str = "all", asr=None) -> TestClient:
    settings = Settings(
        service_name=service_name,
        common=CommonConfig(
            api_key="test-secret",
            vad_rms_threshold=500,
            vad_padding_ms=20,
        ),
    )
    app = create_app(
        settings,
        asr_backend=asr or RecordingASR(),
        qwen_tts_backend=ChunkTTS(),
        cosyvoice_tts_backend=ChunkTTS(),
        kokoro_tts_backend=ChunkTTS(),
    )
    return TestClient(app)


def test_asr_protocol_runs_vad_inside_asr_service() -> None:
    asr = RecordingASR()
    client = make_client(asr=asr)
    with client.websocket_connect(
        "/asr/v1/realtime", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        assert websocket.receive_json() == {
            "type": "session.created",
            "audio_format": "pcm_s16le",
            "sample_rate": 16_000,
            "channels": 1,
        }
        websocket.send_json({"type": "session.update", "language": "ja"})
        websocket.send_bytes(make_audio(0, 4) + make_audio(2_000, 8) + make_audio(0, 4))
        websocket.send_json({"type": "input_audio.commit"})

        result = websocket.receive_json()

    assert result == {
        "type": "transcription.completed",
        "text": "recognized speech",
        "language": "ja",
        "segments": [],
        "is_final": True,
    }
    assert len(asr.calls) == 1
    assert asr.calls[0][1] == "ja"
    assert len(asr.calls[0][0]) < len(make_audio(0, 4) + make_audio(2_000, 8) + make_audio(0, 4))


def test_asr_logs_connection_input_output_and_audio_metadata(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "asr-ws.log"
    monkeypatch.setenv("WSS_LOG_FILE", str(log_path))
    asr = RecordingASR()
    settings = Settings(
        service_name="asr",
        common=CommonConfig(api_key="test-secret", vad_rms_threshold=500, vad_padding_ms=20),
    )
    app = create_app(settings, asr_backend=asr)

    with TestClient(app) as client, client.websocket_connect(
        "/asr/v1/realtime", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "session.update", "language": "ja"})
        websocket.send_bytes(make_audio(2_000, 2))
        websocket.send_json({"type": "input_audio.commit"})
        websocket.receive_json()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    events = {record["event"] for record in records}
    assert {"connection.accepted", "input.json", "input.audio", "output.json"} <= events
    audio = next(record for record in records if record["event"] == "input.audio")
    assert audio["bytes"] == len(make_audio(2_000, 2))
    assert all("test-secret" not in line for line in log_path.read_text(encoding="utf-8").splitlines())


def test_silent_asr_commit_returns_empty_transcript_without_model_call() -> None:
    asr = RecordingASR()
    client = make_client(asr=asr)
    with client.websocket_connect(
        "/asr/v1/realtime", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        websocket.receive_json()
        websocket.send_bytes(make_audio(0, 8))
        websocket.send_json({"type": "input_audio.commit"})
        result = websocket.receive_json()

    assert result["type"] == "transcription.completed"
    assert result["text"] == ""
    assert asr.calls == []


def test_asr_service_can_select_silero_onnx_vad(monkeypatch) -> None:
    captured = {}

    class FakeSileroVAD:
        def __init__(self, model_path, **kwargs):
            captured.update(model_path=model_path, **kwargs)

        def filter(self, audio):
            return audio

    monkeypatch.setattr(app_module, "SileroONNXVAD", FakeSileroVAD)
    settings = Settings(
        service_name="asr",
        common=CommonConfig(
            api_key="test-secret",
            vad_backend="silero",
            vad_model_path="/models/silero_vad.onnx",
            vad_frame_ms=32,
            vad_padding_ms=96,
        ),
    )
    asr = RecordingASR()
    app = create_app(settings, asr_backend=asr)

    with TestClient(app) as client, client.websocket_connect(
        "/asr/v1/realtime", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        websocket.receive_json()
        audio = make_audio(1_000, 8)
        websocket.send_bytes(audio)
        websocket.send_json({"type": "input_audio.commit"})
        websocket.receive_json()

    assert captured == {
        "model_path": "/models/silero_vad.onnx",
        "threshold": 0.5,
        "frame_ms": 32,
        "padding_ms": 96,
        "provider": "CPUExecutionProvider",
    }
    assert asr.calls == [(audio, None)]


def test_asr_requires_bearer_authentication() -> None:
    client = make_client()

    with pytest.raises(WebSocketDisconnect) as error, client.websocket_connect("/asr/v1/realtime"):
        pass

    assert error.value.code == 4401


def test_qwen_tts_protocol_streams_pcm_between_started_and_completed() -> None:
    client = make_client()
    with client.websocket_connect(
        "/tts/v1/realtime", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        websocket.send_json(
            {"type": "response.create", "text": "こんにちは", "voice": "speaker", "language": "ja"}
        )
        started = websocket.receive_json()
        audio = websocket.receive_bytes()
        second_audio = websocket.receive_bytes()
        completed = websocket.receive_json()

    assert started == {
        "type": "response.audio.started",
        "format": "pcm_s16le",
        "sample_rate": 24_000,
        "channels": 1,
    }
    assert len(audio) == len(second_audio) == 512
    assert completed == {"type": "response.completed"}


def test_cosyvoice_protocol_accepts_incremental_text_and_streams_before_end() -> None:
    client = make_client()
    with client.websocket_connect(
        "/tts/cosyvoice", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        websocket.send_json({"type": "start"})
        audio_format = websocket.receive_json()
        websocket.send_json({"type": "text", "text": "第一句。"})
        first_audio = [websocket.receive_bytes(), websocket.receive_bytes()]
        websocket.send_json({"type": "text", "text": "第二句。"})
        second_audio = [websocket.receive_bytes(), websocket.receive_bytes()]
        websocket.send_json({"type": "end"})
        done = websocket.receive_json()

    assert audio_format == {
        "type": "audio_format",
        "sample_rate": 24_000,
        "channels": 1,
        "format": "pcm_s16le",
    }
    assert all(first_audio) and all(second_audio)
    assert done == {"type": "done"}


def test_kokoro_protocol_streams_pcm_between_started_and_completed() -> None:
    client = make_client()
    with client.websocket_connect(
        "/tts/kokoro", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        websocket.send_json({"type": "response.create", "text": "こんにちは", "language": "ja"})
        started = websocket.receive_json()
        audio = websocket.receive_bytes()
        second_audio = websocket.receive_bytes()
        completed = websocket.receive_json()

    assert started == {
        "type": "response.audio.started",
        "format": "pcm_s16le",
        "sample_rate": 24_000,
        "channels": 1,
    }
    assert len(audio) == len(second_audio) == 512
    assert completed == {"type": "response.completed"}


def test_service_only_exposes_its_configured_websocket_path() -> None:
    client = make_client(service_name="asr")

    with pytest.raises(WebSocketDisconnect) as error, client.websocket_connect(
        "/tts/v1/realtime", headers={"Authorization": "Bearer test-secret"}
    ):
        pass

    assert error.value.code == 4404


def test_healthz_reports_the_configured_asr_protocol() -> None:
    client = make_client(service_name="asr")

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "asr",
        "protocols": ["asr"],
    }


def test_healthz_reports_the_configured_qwen_tts_protocol() -> None:
    client = make_client(service_name="qwen-tts")

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "qwen-tts",
        "protocols": ["qwen-tts"],
    }


def test_app_closes_http_backend_resources_on_shutdown() -> None:
    class ClosableTTS(ChunkTTS):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    backend = ClosableTTS()
    settings = Settings(
        service_name="qwen-tts",
        common=CommonConfig(api_key="test-secret"),
    )
    app = create_app(settings, qwen_tts_backend=backend)

    with TestClient(app) as client, client.websocket_connect(
        "/tts/v1/realtime", headers={"Authorization": "Bearer test-secret"}
    ) as websocket:
        websocket.send_json({"type": "response.create", "text": "hello"})
        websocket.receive_json()
        websocket.receive_bytes()
        websocket.receive_bytes()
        websocket.receive_json()

    assert backend.closed is True
