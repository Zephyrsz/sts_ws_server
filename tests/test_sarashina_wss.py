from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import asyncio
import struct

import numpy as np
import pytest

from sts_ws_server.sarashina_wss import create_app
from sts_ws_server.sarashina_wss import SarashinaBackend


class FakeSarashinaBackend:
    sample_rate = 24_000
    ready = True
    not_ready_reason = ""

    def __init__(self, audio: bytes = b"\x01\x00" * 8) -> None:
        self.audio = audio
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio


def test_sarashina_wss_authenticates_and_streams_pcm16_audio() -> None:
    backend = FakeSarashinaBackend()
    client = TestClient(create_app(api_key="secret", backend=backend, chunk_bytes=8))

    with client.websocket_connect(
        "/tts/sarashina", headers={"Authorization": "Bearer secret"}
    ) as websocket:
        websocket.send_json({"type": "response.create", "text": "こんにちは。", "language": "ja"})
        started = websocket.receive_json()
        first_audio = websocket.receive_bytes()
        second_audio = websocket.receive_bytes()
        completed = websocket.receive_json()

    assert started == {
        "type": "response.audio.started",
        "format": "pcm_s16le",
        "sample_rate": 24_000,
        "channels": 1,
    }
    assert first_audio == second_audio == b"\x01\x00" * 4
    assert completed == {"type": "response.completed"}
    assert backend.calls == ["こんにちは。"]


def test_sarashina_wss_rejects_missing_bearer_authentication() -> None:
    client = TestClient(create_app(api_key="secret", backend=FakeSarashinaBackend()))

    with pytest.raises(WebSocketDisconnect) as error, client.websocket_connect(
        "/tts/sarashina"
    ):
        pass

    assert error.value.code == 4401


def test_sarashina_can_synthesize_without_zero_shot_reference_prompt() -> None:
    class FakeGenerator:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [np.asarray([0.5, -0.5], dtype=np.float32)]

    generator = FakeGenerator()
    backend = SarashinaBackend(generator)

    audio = asyncio.run(backend.synthesize("こんにちは。"))

    assert backend.ready
    assert audio == struct.pack("<hh", 16384, -16384)
    assert generator.calls[0]["texts"] == ["こんにちは。"]
    assert generator.calls[0]["flow_embedding"] is None
    assert generator.calls[0]["audio_prompt_text"] is None
    assert generator.calls[0]["audio_prompt_tokens"] is None
    assert generator.calls[0]["audio_prompt_feat"] is None
    assert generator.calls[0]["audio_prompt_path"] is None


def test_sarashina_health_does_not_claim_ready_when_model_is_unavailable() -> None:
    backend = FakeSarashinaBackend()
    backend.ready = False
    backend.not_ready_reason = "Sarashina model is not loaded"
    client = TestClient(create_app(api_key="secret", backend=backend))

    assert client.get("/healthz").json() == {
        "status": "not_ready",
        "service": "sarashina",
        "protocols": ["sarashina"],
        "reason": "Sarashina model is not loaded",
    }

    with client.websocket_connect(
        "/tts/sarashina", headers={"Authorization": "Bearer secret"}
    ) as websocket:
        websocket.send_json({"type": "response.create", "text": "こんにちは。"})
        assert websocket.receive_json() == {
            "type": "error",
            "message": "Sarashina model is not ready",
        }

    assert backend.calls == []
