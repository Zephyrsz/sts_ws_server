from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).parents[1] / "demo1" / "test_public_wss.py"
_SPEC = importlib.util.spec_from_file_location("test_public_wss_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

chunk_pcm = _MODULE.chunk_pcm
parse_json_frame = _MODULE.parse_json_frame
parse_llm_frame = _MODULE.parse_llm_frame

"""Protocol-only tests for the public WSS smoke-test helpers."""


def test_chunk_pcm_keeps_even_pcm16_frames_and_counts_bytes() -> None:
    chunks = list(chunk_pcm(b"0123456789", chunk_bytes=4))

    assert chunks == [b"0123", b"4567", b"89"]


def test_chunk_pcm_rejects_odd_pcm16_payload() -> None:
    try:
        list(chunk_pcm(b"123", chunk_bytes=4))
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("odd PCM16 payload must be rejected")


def test_parse_json_frame_rejects_binary_and_non_object_messages() -> None:
    assert parse_json_frame('{"type":"session.created"}') == {
        "type": "session.created"
    }

    for message in (b"audio", json.dumps(["not", "an", "object"])):
        try:
            parse_json_frame(message)
        except ValueError as exc:
            assert "JSON object" in str(exc)
        else:
            raise AssertionError("invalid websocket frame must be rejected")


def test_parse_llm_frame_extracts_delta_content_and_done() -> None:
    frame = json.dumps(
        {"choices": [{"delta": {"content": "OK"}, "finish_reason": None}]}
    )

    assert parse_llm_frame(frame) == (False, "OK")
    assert parse_llm_frame("[DONE]") == (True, "")


def test_parse_llm_frame_does_not_return_upstream_error_body() -> None:
    try:
        parse_llm_frame(json.dumps({"error": {"message": "private diagnostic"}}))
    except RuntimeError as exc:
        assert str(exc) == "LLM server returned an error"
    else:
        raise AssertionError("LLM error frame must fail the smoke test")


def test_tls_failure_explains_ca_certificate_mismatch() -> None:
    error = _MODULE.tls_error_message(
        "wss://47.87.138.166:18443/ws/v1/chat/completions",
        "/tmp/llm-wss.crt",
    )

    assert "18443" in error
    assert "/tmp/llm-wss.crt" in error
    assert "--insecure" in error


class FakeWebSocket:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.messages = iter(messages)
        self.sent: list[tuple[object, dict[str, object]]] = []
        self.closed = False

    def recv(self) -> str | bytes:
        return next(self.messages)

    def send(self, payload: object, **kwargs: object) -> None:
        self.sent.append((payload, kwargs))

    def close(self) -> None:
        self.closed = True


def _run_with_fake_socket(monkeypatch, messages, runner, **kwargs):
    socket = FakeWebSocket(messages)
    monkeypatch.setattr(_MODULE, "make_connection", lambda *args, **options: socket)
    result = runner(
        base_url="wss://example",
        api_key="secret",
        ssl_options={},
        timeout=1,
        **kwargs,
    )
    assert result.passed, result.error
    assert socket.closed
    return socket


def test_asr_runner_rejects_wrong_session_format(monkeypatch, tmp_path) -> None:
    pcm_path = tmp_path / "input.pcm"
    pcm_path.write_bytes(b"0" * 640)
    socket = FakeWebSocket(
        [
            '{"type":"session.created","audio_format":"pcm_f32le","sample_rate":16000,"channels":1}',
        ]
    )
    monkeypatch.setattr(_MODULE, "make_connection", lambda *args, **options: socket)

    result = _MODULE.run_asr(
        base_url="wss://example",
        api_key="secret",
        ssl_options={},
        timeout=1,
        pcm_path=pcm_path,
        language="ja",
        chunk_bytes=640,
    )

    assert not result.passed
    assert result.error == "invalid ASR session format"


def test_service_runners_complete_successfully_with_protocol_frames(monkeypatch, tmp_path) -> None:
    pcm_path = tmp_path / "input.pcm"
    pcm_path.write_bytes(b"0" * 640)

    asr_socket = _run_with_fake_socket(
        monkeypatch,
        [
            '{"type":"session.created","audio_format":"pcm_s16le","sample_rate":16000,"channels":1}',
            '{"type":"transcription.completed","text":"ok"}',
        ],
        _MODULE.run_asr,
        pcm_path=pcm_path,
        language="ja",
        chunk_bytes=640,
    )
    assert json.loads(asr_socket.sent[-1][0]) == {"type": "input_audio.commit"}

    _run_with_fake_socket(
        monkeypatch,
        [
            '{"type":"response.audio.started","format":"pcm_s16le","sample_rate":24000,"channels":1}',
            b"\\x00\\x00",
            '{"type":"response.completed"}',
        ],
        _MODULE.run_qwen_tts,
        text="hi",
        voice="Ono_Anna",
        language="ja",
        output_path=None,
    )
    _run_with_fake_socket(
        monkeypatch,
        [
            '{"type":"audio_format","format":"pcm_s16le","sample_rate":24000,"channels":1}',
            b"\\x00\\x00",
            '{"type":"done"}',
        ],
        _MODULE.run_cosyvoice,
        text_chunks=("hi",),
        output_path=None,
    )
    _run_with_fake_socket(
        monkeypatch,
        ['{"choices":[{"delta":{"content":"OK"}}]}', "[DONE]"],
        _MODULE.run_llm,
        model="Ornith",
        prompt="Reply OK",
    )
