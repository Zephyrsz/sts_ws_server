# WSS Independent Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent JSONL logs for all five WSS services, recording connection, text/JSON input/output, and binary audio metadata with daily 14-day rotation, then deploy to `pj_gpuserver2`.

**Architecture:** A standard-library `ServiceEventLogger` owns one JSONL `TimedRotatingFileHandler` per process and is configured by `WSS_LOG_FILE`. The shared app, LLM bridge, and Sarashina app emit structured events at WebSocket boundaries; systemd units create the log directory and assign distinct files. Audio payloads are represented by counts and frame metadata only.

**Tech Stack:** Python 3.10+, FastAPI/Starlette WebSockets, `logging.handlers.TimedRotatingFileHandler`, systemd `LogsDirectory`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-wss-independent-logging-design.md`

## Global Constraints

- Do not log bearer/API key values or binary audio payloads.
- Keep existing journal output enabled for startup and exception diagnostics.
- Retain 14 daily file backups per service.
- Preserve the repository's existing untracked `voice_example/sample/` files.
- Deploy only through `/Users/rgwei/pj/pj_agent/pj_gpusever2/remote.py` and its configured SSH target.

### Task 1: Structured logger

**Files:**
- Create: `src/sts_ws_server/logging.py`
- Test: `tests/test_logging.py`

**Interfaces:**
- Produces `ServiceEventLogger(service_name, log_path=None, backup_count=14)` with `.event(name, **fields)` and `.close()`.
- Consumes `WSS_LOG_FILE` and `WSS_LOG_BACKUP_COUNT` when explicit arguments are absent.

- [ ] **Step 1: Write failing tests** for JSONL output, UTC timestamp/service/event fields, secret omission, binary metadata, and `backupCount == 14`.
- [ ] **Step 2: Run `pytest tests/test_logging.py -q` and verify the new import/API fails.**
- [ ] **Step 3: Implement the logger with a JSON formatter, safe field filtering, delayed timed rotation, and no-op behavior when no path is configured.**
- [ ] **Step 4: Run `pytest tests/test_logging.py -q` and verify all logger tests pass.**

### Task 2: Shared ASR/TTS events

**Files:**
- Modify: `src/sts_ws_server/app.py`
- Test: `tests/test_websocket_protocol.py`

**Interfaces:**
- Consumes `ServiceEventLogger` from Task 1.
- Emits `connection.accepted`, `connection.rejected`, `connection.closed`, `input.json`, `input.audio`, `output.json`, `output.audio`, and `request.error`.

- [ ] **Step 1: Add tests that create an app with a temporary `WSS_LOG_FILE` and assert ASR, Qwen TTS, and Kokoro event records.**
- [ ] **Step 2: Run the focused tests and verify they fail because no event records exist.**
- [ ] **Step 3: Add connection IDs, safe client metadata, JSON input/output records, aggregate audio byte/frame counts, and request timing to all shared routes.**
- [ ] **Step 4: Run the focused protocol and logger tests.**

### Task 3: LLM and Sarashina events

**Files:**
- Modify: `src/sts_ws_server/llm_wss.py`
- Modify: `src/sts_ws_server/sarashina_wss.py`
- Test: `tests/test_llm_wss.py`
- Test: `tests/test_sarashina_wss.py`

**Interfaces:**
- Consumes the same logger API from Task 1.
- Records parsed LLM request JSON and streamed output text; records Sarashina request JSON and aggregate PCM metadata.

- [ ] **Step 1: Add failing event assertions to representative LLM and Sarashina WebSocket tests.**
- [ ] **Step 2: Run the focused tests and verify missing event records.**
- [ ] **Step 3: Instrument auth, accept/disconnect, JSON request/response, streamed output, audio totals, and error paths.**
- [ ] **Step 4: Run all WebSocket tests and verify protocol behavior is unchanged.**

### Task 4: Deployment configuration

**Files:**
- Modify: `deploy/asr-ws.service`
- Modify: `deploy/sts-ws-qwen.service`
- Modify: `deploy/kokoro-ws.service`
- Modify: `deploy/sarashina-ws.service`
- Modify: `deploy/ornith/llm-wss.service`

**Interfaces:**
- Produces `/var/log/sts-ws-server/<service>.log` for each WSS unit.
- Keeps systemd journal output active.

- [ ] **Step 1: Add `LogsDirectory`, write access, and distinct `WSS_LOG_FILE` values to each unit.**
- [ ] **Step 2: Validate unit syntax locally with a textual check and run the full test suite.**

### Task 5: Remote deployment and verification

**Files:**
- Deploy changed source files to `/app/sts_ws_server/src/sts_ws_server`, `/app/sts_ws_server/service-src/src/sts_ws_server`, and `/app/sts_ws_server/asr-service/src/sts_ws_server`.
- Deploy unit files to `/etc/systemd/system` and logrotate policy to `/etc/logrotate.d`.

- [ ] **Step 1: Preflight remote hostname/user/path with `remote.py`.**
- [ ] **Step 2: Copy files using `remote.py`/its Python copy API, preserving secrets and models.**
- [ ] **Step 3: Run `systemctl daemon-reload` and restart the five WSS units one at a time.**
- [ ] **Step 4: Verify ports, `/healthz`, service status, log file ownership, and a live event in each log.**
