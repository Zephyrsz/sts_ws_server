from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import asyncio
import struct
from pathlib import Path

import numpy as np
import pytest

from sts_ws_server.sarashina_wss import (
    SarashinaBackend,
    SarashinaVoiceProfile,
    UnknownSarashinaVoice,
    create_app,
    load_voice_profiles,
)


class FakeSarashinaBackend:
    sample_rate = 24_000
    ready = True
    not_ready_reason = ""

    def __init__(self, audio: bytes = b"\x01\x00" * 8) -> None:
        self.audio = audio
        self.calls: list[tuple[str, str | None]] = []

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        self.calls.append((text, voice))
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
    assert backend.calls == [("こんにちは。", None)]


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


def test_sarashina_backend_selects_cached_voice_conditioning(tmp_path) -> None:
    class FakeGenerator:
        def __init__(self) -> None:
            self.extract_calls: list[tuple[str, str]] = []
            self.calls: list[dict] = []

        def _extract_zero_shot_embedding(self, path: str):
            self.extract_calls.append(("embedding", path))
            return f"embedding:{path}"

        def _extract_audio_prompt_tokens(self, path: str):
            self.extract_calls.append(("tokens", path))
            return [path]

        def _extract_audio_prompt_feat(self, path: str):
            self.extract_calls.append(("feat", path))
            return f"feat:{path}"

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [np.asarray([0.25], dtype=np.float32)]

    generator = FakeGenerator()
    first = str(tmp_path / "sarashina-sakura.wav")
    second = str(tmp_path / "sarashina-yuki.wav")
    Path(first).touch()
    Path(second).touch()
    backend = SarashinaBackend(
        generator,
        voice_profiles={
            "sakura": SarashinaVoiceProfile(first, "桜の声です。"),
            "yuki": SarashinaVoiceProfile(second, "雪の声です。"),
        },
        default_voice="sakura",
    )

    assert backend.ready
    assert backend.available_voices == ("sakura", "yuki")
    assert backend.default_voice == "sakura"
    assert len(generator.extract_calls) == 6

    asyncio.run(backend.synthesize("こんにちは。", voice="yuki"))
    asyncio.run(backend.synthesize("おはようございます。", voice="yuki"))

    call = generator.calls[0]
    assert call["flow_embedding"] == f"embedding:{second}"
    assert call["audio_prompt_text"] == "雪の声です。"
    assert call["audio_prompt_tokens"] == [second]
    assert call["audio_prompt_feat"] == f"feat:{second}"
    assert call["audio_prompt_path"] == second
    assert len(generator.extract_calls) == 6
    assert len(generator.calls) == 2

    with pytest.raises(UnknownSarashinaVoice, match="unknown Sarashina voice: robot"):
        asyncio.run(backend.synthesize("こんにちは。", voice="robot"))


def test_sarashina_wss_routes_voice_and_rejects_unknown_voice() -> None:
    class VoiceBackend(FakeSarashinaBackend):
        available_voices = ("sakura", "yuki")
        default_voice = "sakura"

        async def synthesize(self, text: str, voice: str | None = None) -> bytes:
            if voice not in {None, *self.available_voices}:
                raise UnknownSarashinaVoice(f"unknown Sarashina voice: {voice}")
            return await super().synthesize(text, voice)

        def resolve_voice(self, voice: str | None) -> str:
            selected = voice or self.default_voice
            if selected not in self.available_voices:
                raise UnknownSarashinaVoice(f"unknown Sarashina voice: {selected}")
            return selected

    backend = VoiceBackend()
    client = TestClient(create_app(api_key="secret", backend=backend, chunk_bytes=8))

    with client.websocket_connect(
        "/tts/sarashina", headers={"Authorization": "Bearer secret"}
    ) as websocket:
        websocket.send_json(
            {"type": "response.create", "text": "こんにちは。", "voice": "yuki"}
        )
        assert websocket.receive_json()["type"] == "response.audio.started"
        assert websocket.receive_bytes() == b"\x01\x00" * 4
        assert websocket.receive_bytes() == b"\x01\x00" * 4
        assert websocket.receive_json() == {"type": "response.completed"}

        websocket.send_json({"type": "response.create", "text": "おはよう。"})
        assert websocket.receive_json()["type"] == "response.audio.started"
        assert websocket.receive_bytes() == b"\x01\x00" * 4
        assert websocket.receive_bytes() == b"\x01\x00" * 4
        assert websocket.receive_json() == {"type": "response.completed"}

        websocket.send_json(
            {"type": "response.create", "text": "こんにちは。", "voice": "robot"}
        )
        assert websocket.receive_json() == {
            "type": "error",
            "message": "unknown Sarashina voice: robot",
        }

    assert backend.calls == [("こんにちは。", "yuki"), ("おはよう。", "sakura")]


def test_load_voice_profiles_resolves_relative_audio_paths(tmp_path) -> None:
    config = tmp_path / "voices.json"
    config.write_text(
        '{"default_voice":"sakura","voices":{"sakura":{"audio":"audio/0.wav",'
        '"text":"桜の声です。","label":"温柔樱"}}}',
        encoding="utf-8",
    )

    default_voice, profiles = load_voice_profiles(config)

    assert default_voice == "sakura"
    assert profiles == {
        "sakura": SarashinaVoiceProfile(
            str(tmp_path / "audio/0.wav"), "桜の声です。", "温柔樱"
        )
    }


def test_repository_voice_profiles_define_three_sarashina_voices() -> None:
    config = Path(__file__).parents[1] / "deploy/sarashina-voices.example.json"

    default_voice, profiles = load_voice_profiles(config)

    assert default_voice == "sakura"
    assert set(profiles) == {"sakura", "tsubaki", "yuki"}
    assert all(Path(profile.audio_path).is_file() for profile in profiles.values())
    assert profiles["sakura"].text.startswith("底の角を右に曲がって")
    assert profiles["tsubaki"].text.startswith("南の空で一番明るく")
    assert profiles["yuki"].text.startswith("お湯加減はいかがですか")


def test_load_voice_profiles_requires_a_configured_default(tmp_path) -> None:
    config = tmp_path / "voices.json"
    config.write_text(
        '{"default_voice":"missing","voices":{"sakura":'
        '{"audio":"voice.wav","text":"桜の声です。"}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_voice must name a configured voice"):
        load_voice_profiles(config)
