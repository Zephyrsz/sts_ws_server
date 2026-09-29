from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "refresh_token",
    "secret",
    "token",
    "upstream_api_key",
    "wss_api_key",
}


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith("_token")


def _json_safe(value: Any, *, key: object | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return "[redacted]"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(item_key): _json_safe(item_value, key=item_key) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return {"type": "binary", "bytes": len(value)}
    return str(value)


class _JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": self.service_name,
            "event": getattr(record, "event", record.getMessage()),
        }
        fields = getattr(record, "fields", {})
        if isinstance(fields, Mapping):
            for key, value in fields.items():
                if key not in payload:
                    payload[str(key)] = _json_safe(value, key=key)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ServiceEventLogger:
    """Write service-scoped JSONL events without exposing authentication secrets."""

    def __init__(
        self,
        service_name: str,
        log_path: str | Path | None = None,
        backup_count: int | None = None,
    ) -> None:
        self.service_name = service_name
        configured_path = log_path or os.getenv("WSS_LOG_FILE")
        configured_backups = backup_count
        if configured_backups is None:
            configured_backups = int(os.getenv("WSS_LOG_BACKUP_COUNT", "14"))
        if configured_backups < 1:
            raise ValueError("backup_count must be positive")

        self.handler: TimedRotatingFileHandler | None = None
        logger_name = f"sts_ws_server.events.{service_name}.{id(self)}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if configured_path:
            try:
                self.handler = TimedRotatingFileHandler(
                    Path(configured_path),
                    when="midnight",
                    interval=1,
                    backupCount=configured_backups,
                    encoding="utf-8",
                    delay=True,
                    utc=True,
                )
                self.handler.setFormatter(_JsonFormatter(service_name))
                self._logger.addHandler(self.handler)
            except (OSError, ValueError):
                self.handler = None

    @classmethod
    def from_environment(cls, service_name: str) -> ServiceEventLogger:
        return cls(service_name)

    def event(self, event_name: str, **fields: Any) -> None:
        if self.handler is None:
            return
        try:
            self._logger.info(
                event_name,
                extra={"event": event_name, "fields": fields},
            )
        except Exception:
            # File logging must never take down an active WebSocket request.
            return

    def close(self) -> None:
        if self.handler is None:
            return
        self._logger.removeHandler(self.handler)
        self.handler.close()
