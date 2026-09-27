#!/usr/bin/env python3
"""Named entry point for the server-2 public streaming LLM WSS test."""

from __future__ import annotations

from test1 import (  # noqa: F401 - keep the existing test1 CLI as the implementation
    DEFAULT_BASE_URL,
    DEFAULT_PROMPT,
    build_parser,
    main,
    parse_frame,
    ssl_options,
)


if __name__ == "__main__":
    raise SystemExit(main())
