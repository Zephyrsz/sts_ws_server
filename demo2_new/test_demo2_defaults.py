from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_public_test_uses_server_two_defaults_and_separate_auth(monkeypatch):
    for variable in ("STS2_WSS_BASE_URL", "STS2_API_KEY", "STS2_CA_CERT", "STS2_LLM_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("MIMO_WSS_BASE_URL", "wss://47.87.138.166:18443")
    monkeypatch.setenv("MIMO_API_KEY", "server-one-key")
    monkeypatch.setenv("MIMO_CA_CERT", "/server-one.crt")

    module = load_script("test_public_wss")
    args = module.build_parser().parse_args([])

    assert args.base_url == "wss://8.211.122.183:18443"
    assert args.api_key is None
    assert args.ca_cert == str(ROOT / "server2-wss.crt")
    assert args.llm_model == "Ornith-1.5-35B-A3B"


def test_llm_test_uses_server_two_defaults_and_separate_auth(monkeypatch):
    for variable in ("STS2_WSS_BASE_URL", "STS2_API_KEY", "STS2_CA_CERT", "STS2_LLM_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("MIMO_WSS_BASE_URL", "wss://47.87.138.166:18443")
    monkeypatch.setenv("MIMO_API_KEY", "server-one-key")
    monkeypatch.setenv("MIMO_CA_CERT", "/server-one.crt")

    module = load_script("test1")
    args = module.build_parser().parse_args([])

    assert args.base_url == "wss://8.211.122.183:18443"
    assert args.api_key is None
    assert args.ca_cert == str(ROOT / "server2-wss.crt")
    assert args.model == "Ornith-1.5-35B-A3B"
