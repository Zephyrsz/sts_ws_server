from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).parents[1]
DEMO2 = ROOT / "demo2"
if str(DEMO2) not in sys.path:
    sys.path.insert(0, str(DEMO2))


def load_script(name: str):
    path = DEMO2 / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_asr_script_defaults_and_pcm_validation(tmp_path):
    module = load_script("test_asr_wss")
    args = module.build_parser().parse_args(["--pcm", str(tmp_path / "speech.pcm")])
    assert args.base_url == "wss://8.211.122.183:18443"
    assert args.ca_cert == DEMO2 / "server2-wss.crt"
    assert args.language == "ja"
    assert args.chunk_bytes == 640

    pcm = tmp_path / "speech.pcm"
    pcm.write_bytes(b"\x00\x00" * 4)
    assert module.load_pcm16(pcm) == b"\x00\x00" * 4

    wav = tmp_path / "speech.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x01\x00" * 4)
    assert module.load_pcm16(wav) == b"\x01\x00" * 4


def test_asr_script_sends_session_update_audio_and_commit(monkeypatch, tmp_path):
    module = load_script("test_asr_wss")
    pcm = tmp_path / "speech.pcm"
    pcm.write_bytes(b"\x00\x00" * 4)

    class FakeWebSocket:
        def __init__(self):
            self.messages = iter(
                [
                    json.dumps(
                        {
                            "type": "session.created",
                            "audio_format": "pcm_s16le",
                            "sample_rate": 16_000,
                            "channels": 1,
                        }
                    ),
                    json.dumps(
                        {
                            "type": "transcription.completed",
                            "text": "测试文本",
                            "language": "zh",
                        }
                    ),
                ]
            )
            self.sent = []
            self.closed = False

        def recv(self):
            return next(self.messages)

        def send(self, payload, **kwargs):
            self.sent.append((payload, kwargs))

        def close(self):
            self.closed = True

    socket = FakeWebSocket()
    monkeypatch.setattr(module, "make_connection", lambda *args, **kwargs: socket)
    args = module.build_parser().parse_args(
        ["--pcm", str(pcm), "--api-key", "secret", "--insecure"]
    )

    assert module.run_asr(args) == 0
    assert socket.closed
    assert json.loads(socket.sent[0][0]) == {"type": "session.update", "language": "ja"}
    assert socket.sent[1][0] == b"\x00\x00" * 4
    assert socket.sent[1][1]["opcode"] == module._require_websocket_client().ABNF.OPCODE_BINARY
    assert json.loads(socket.sent[2][0]) == {"type": "input_audio.commit"}


def test_llm_script_uses_server_two_defaults_and_parses_stream():
    module = load_script("test_llm_wss")
    args = module.build_parser().parse_args([])
    assert args.base_url == "wss://8.211.122.183:18443"
    assert args.ca_cert == str(DEMO2 / "server2-wss.crt")
    assert args.model == "Ornith-1.5-35B-A3B"
    assert module.parse_frame('{"choices":[{"delta":{"content":"OK"}}]}') == (False, "OK")
    assert module.parse_frame("[DONE]") == (True, "")
