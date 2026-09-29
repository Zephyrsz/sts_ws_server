import json

from sts_ws_server.logging import ServiceEventLogger


def test_service_event_logger_writes_structured_jsonl_without_secrets(tmp_path) -> None:
    path = tmp_path / "asr-ws.log"
    logger = ServiceEventLogger("asr", path)

    logger.event(
        "connection.accepted",
        connection_id="conn-1",
        client_ip="127.0.0.1",
        path="/asr/v1/realtime",
        authorization="Bearer should-not-be-written",
        payload={"text": "hello", "api_key": "nested-secret"},
    )
    logger.event("input.audio", connection_id="conn-1", bytes=128, frames=2)
    logger.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["service"] == "asr"
    assert records[0]["event"] == "connection.accepted"
    assert records[0]["connection_id"] == "conn-1"
    assert records[0]["payload"]["text"] == "hello"
    assert records[0]["authorization"] == "[redacted]"
    assert records[0]["payload"]["api_key"] == "[redacted]"
    assert records[1]["event"] == "input.audio"
    assert records[1]["bytes"] == 128
    assert records[1]["frames"] == 2
    assert "should-not-be-written" not in path.read_text(encoding="utf-8")
    assert "nested-secret" not in path.read_text(encoding="utf-8")


def test_service_event_logger_uses_fourteen_daily_backups(tmp_path) -> None:
    logger = ServiceEventLogger("llm-wss", tmp_path / "llm-wss.log")

    assert logger.handler is not None
    assert logger.handler.when == "MIDNIGHT"
    assert logger.handler.backupCount == 14

    logger.close()
