# WSS Independent Logging Design

## Goal

Give each WebSocket service its own JSONL application log that records connection events, text/JSON input and output, and metadata for binary audio without writing audio payloads to disk.

## Design

The application will use a small standard-library logging adapter in `src/sts_ws_server/logging.py`. Each process receives a service-specific `WSS_LOG_FILE` from its systemd unit. The adapter writes structured JSON lines with a UTC timestamp, service name, event name, connection ID, client address, path, and event-specific fields. `TimedRotatingFileHandler` rotates at midnight and retains 14 backups.

The shared ASR/TTS app will create one event logger from `settings.service_name`; the LLM and Sarashina apps will create their own named loggers. Connection acceptance/rejection and disconnects are logged for every WebSocket. JSON messages are recorded as parsed payloads. Audio input and output are recorded as byte counts, frame counts, format, and duration-related metadata; PCM/WAV bytes are never serialized into logs. Existing systemd journal output remains enabled as a fallback for startup and exception diagnostics.

Systemd units will create `/var/log/sts-ws-server` with `LogsDirectory`, set a distinct log filename per service, and expose the 14-backup setting. Deployment will update all three installed source copies used by ASR, shared TTS, and LLM/Sarashina services, reload systemd, restart the five WSS units, and verify health and log writes.

## Operational behavior

- Files: `asr-ws.log`, `qwen-tts-ws.log`, `kokoro-ws.log`, `sarashina-ws.log`, and `llm-wss.log` under `/var/log/sts-ws-server`.
- File permissions: created by the service user through systemd's `LogsDirectory`; handlers use UTF-8 and append mode.
- Retention: application-level daily rotation with 14 backup files per service; a second external rotator is intentionally not configured so two rotation policies cannot race.
- Secrets: authorization headers and API keys are never logged; only an `authorized` boolean is recorded.
- Failure mode: a file logging failure is caught by the adapter and does not fail a WebSocket request; normal Python/systemd logging continues.

## Verification

Unit tests will cover JSONL fields, secret omission, audio metadata, and rotation configuration. WebSocket protocol tests will assert representative connection, input, and output events. Remote verification will check all five files, unit health, and a live connection event.
