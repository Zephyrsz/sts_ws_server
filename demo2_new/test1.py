#!/usr/bin/env python3
"""Send a normal Chat Completions request and print the streamed answer."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any

try:
    import websocket
except ModuleNotFoundError:
    raise SystemExit(
        "Missing websocket-client. Run with the project interpreter, for example: "
        ".venv/bin/python demo2/test1.py"
    ) from None


DEFAULT_BASE_URL = "wss://8.211.122.183:18443"
DEFAULT_CA_CERT = str(Path(__file__).with_name("server2-wss.crt"))
DEFAULT_PROMPT = "请用一句话介绍你自己。"


def parse_frame(message: str | bytes) -> tuple[bool, str]:
    """Return ``(done, text_delta)`` for one bridge response frame."""

    if isinstance(message, bytes):
        raise RuntimeError("LLM bridge returned a binary frame")
    if message == "[DONE]":
        return True, ""
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM bridge returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LLM bridge returned a non-object JSON frame")
    if "error" in payload:
        error = payload.get("error")
        detail = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(f"LLM request failed: {detail or 'unknown error'}")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM frame has no choices")
    choice = choices[0]
    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
    if not isinstance(delta, dict):
        raise RuntimeError("LLM frame has an invalid delta")
    content = delta.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise RuntimeError("LLM content delta is not text")
    return False, content


def ssl_options(ca_cert: str | None, insecure: bool) -> dict[str, Any]:
    if insecure:
        return {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
    options: dict[str, Any] = {"cert_reqs": ssl.CERT_REQUIRED}
    if ca_cert:
        options["ca_certs"] = ca_cert
    return options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("STS2_WSS_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("STS2_API_KEY"))
    parser.add_argument("--ca-cert", default=os.getenv("STS2_CA_CERT", DEFAULT_CA_CERT))
    parser.add_argument("--insecure", action="store_true", help="disable TLS verification")
    parser.add_argument("--model", default=os.getenv("STS2_LLM_MODEL", "Ornith-1.5-35B-A3B"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="user message")
    parser.add_argument("--system", default="", help="optional system message")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.api_key:
        raise SystemExit("Set STS2_API_KEY or pass --api-key.")
    if args.max_tokens <= 0 or args.timeout <= 0:
        raise SystemExit("--max-tokens and --timeout must be positive.")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    request = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{args.base_url.rstrip('/')}/ws/v1/chat/completions"
    print(f"Connecting: {url}")
    print(f"Prompt: {args.prompt}")
    if args.insecure:
        print("TLS verification: disabled (--insecure)")
    elif args.ca_cert:
        print(f"TLS verification: enabled with CA {args.ca_cert}")
    else:
        print("TLS verification: enabled with system CA store")

    started = time.perf_counter()
    ws = websocket.create_connection(
        url,
        header=[f"Authorization: Bearer {args.api_key}"],
        sslopt=ssl_options(args.ca_cert, args.insecure),
        timeout=args.timeout,
    )
    first_frame = None
    chunks = 0
    characters = 0
    try:
        ws.send(json.dumps(request, ensure_ascii=False))
        print("Assistant: ", end="", flush=True)
        while True:
            message = ws.recv()
            if first_frame is None:
                first_frame = time.perf_counter() - started
            done, content = parse_frame(message)
            if content:
                print(content, end="", flush=True)
                chunks += 1
                characters += len(content)
            if done:
                break
        print()
    except ssl.SSLCertVerificationError as exc:
        raise SystemExit(
            f"TLS verification failed for {url}; check STS2_CA_CERT or use --insecure for diagnosis: {exc}"
        ) from None
    finally:
        ws.close()

    print(
        f"Stream complete: chunks={chunks}, characters={characters}, "
        f"first_frame={(first_frame or 0.0):.3f}s"
    )
    return 0 if characters else 1


if __name__ == "__main__":
    sys.exit(main())
