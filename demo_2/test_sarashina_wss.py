#!/usr/bin/env python3
"""Convert Japanese text to a WAV file through server 2 Sarashina WSS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tts_wss_common import DEFAULT_TEXT, add_common_arguments, run_response_create


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(text=DEFAULT_TEXT, output=Path("demo2/sarashina-ja.wav"))
    parser.add_argument("--language", default="ja", help="Language passed to the adapter")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_response_create(
        service_name="sarashina",
        endpoint="/tts/sarashina",
        base_url=args.base_url,
        api_key=args.api_key,
        ca_cert=args.ca_cert,
        insecure=args.insecure,
        timeout=args.timeout,
        request={"type": "response.create", "text": args.text, "language": args.language},
        output=args.output,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
