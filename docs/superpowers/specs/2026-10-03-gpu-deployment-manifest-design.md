# GPU Deployment Manifest Design

## Goal

Keep GPU assignments, service instance IDs, and ports for the WebSocket deployment in one local declarative file. The same shape must support one service instance today and multiple Sarashina instances later without copying systemd units by hand.

## Design

`deploy/gpu-services.json` is the source of truth. Each service entry has a stable `id`, a `kind`, one or more physical `gpu_ids`, and optional WebSocket/model ports. The manifest records intentional GPU sharing explicitly because the current host shares GPU 0 between Qwen3-TTS and Kokoro.

`deploy/gpu_deployment.py` loads and validates the manifest using only the Python standard library. Validation checks the schema, service IDs, GPU ranges, port collisions, required ports, and unapproved GPU sharing. It can emit a normalized deployment plan and per-instance environment fragments without contacting systemd or a remote host.

Generated output is disposable and written to a caller-selected directory. Secrets remain in existing environment files and are never copied into the manifest or generated output.

## Future extension

Adding another Sarashina process is one additional service entry with a new ID, GPU, and port. Multi-GPU models use `gpu_ids` with more than one value. A later migration step can consume the generated plan to create systemd units on the target host; this change intentionally does not install or restart any remote service.

## Verification

Tests cover the current pj_gpuserver2 mapping, multi-instance Sarashina entries, invalid GPU/port assignments, and rendered environment contents. The CLI is verified with local `validate`, `plan`, and `render` commands.
