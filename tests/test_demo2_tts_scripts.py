from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "demo2"))


def load_script(name: str):
    path = ROOT / "demo2" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_script_has_server_two_defaults(monkeypatch):
    for variable in ("STS2_WSS_BASE_URL", "STS2_API_KEY", "STS2_CA_CERT"):
        monkeypatch.delenv(variable, raising=False)
    for name, endpoint, output in (
        ("test_qwen_tts_wss", "/tts/v1/realtime", "qwen-tts-ja.wav"),
        ("test_kokoro_wss", "/tts/kokoro", "kokoro-ja.wav"),
        ("test_sarashina_wss", "/tts/sarashina", "sarashina-ja.wav"),
    ):
        module = load_script(name)
        args = module.build_parser().parse_args([])
        assert args.base_url == "wss://8.211.122.183:18443"
        assert args.ca_cert == ROOT / "demo2" / "server2-wss.crt"
        assert str(args.output).endswith(output)
        assert endpoint.startswith("/tts/")


def test_common_helper_writes_pcm16_wave(tmp_path):
    from tts_wss_common import write_wav

    output = tmp_path / "voice.wav"
    write_wav(output, b"\x00\x00" * 4, 24_000, 1)
    assert output.read_bytes().startswith(b"RIFF")


def test_sarashina_script_accepts_a_voice_profile(monkeypatch):
    module = load_script("test_sarashina_wss")
    args = module.build_parser().parse_args(["--voice", "tsubaki"])
    assert args.voice == "tsubaki"
