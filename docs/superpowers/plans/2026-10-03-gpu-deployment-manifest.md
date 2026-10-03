# GPU Deployment Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, dependency-free manifest and renderer for assigning GPU IDs to WebSocket service instances.

**Architecture:** A JSON manifest describes host capacity and service instances. A standard-library loader validates it and emits a normalized plan plus safe per-instance environment files; no systemd command or remote mutation is performed.

**Tech Stack:** Python 3.10+, `dataclasses`, `json`, `argparse`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-03-gpu-deployment-manifest-design.md`

## Global Constraints

- Do not modify the remote `pj_gpuserver2` host.
- Do not include secrets in the manifest or generated output.
- Preserve intentional GPU sharing only when it is explicitly declared.
- Keep the implementation dependency-free at runtime.

---

### Task 1: Define and validate the manifest model

**Files:**
- Create: `deploy/gpu_deployment.py`
- Create: `tests/test_gpu_deployment.py`

**Interfaces:**
- `load_manifest(path: Path) -> DeploymentManifest`
- `DeploymentManifest.validate() -> None`
- `DeploymentManifest.to_plan() -> dict[str, object]`

- [ ] Write tests for the current mapping, multiple Sarashina entries, invalid GPU IDs, duplicate ports, and undeclared GPU sharing.
- [ ] Run `pytest -q tests/test_gpu_deployment.py` and confirm the new tests fail because the loader is absent.
- [ ] Implement dataclasses, JSON loading, schema validation, conflict detection, and normalized plan generation.
- [ ] Re-run the focused test file and confirm it passes.

### Task 2: Add safe local rendering and CLI

**Files:**
- Modify: `deploy/gpu_deployment.py`
- Create: `deploy/gpu-services.json`
- Modify: `tests/test_gpu_deployment.py`

**Interfaces:**
- `render_manifest(manifest: DeploymentManifest, output_dir: Path) -> None`
- CLI commands: `validate`, `plan`, and `render`.

- [ ] Add tests for rendered `plan.json` and per-instance `.env` files, including physical `CUDA_VISIBLE_DEVICES` and logical `cuda:0` mapping.
- [ ] Run the focused tests and confirm the renderer tests fail before implementation.
- [ ] Implement rendering with atomic text writes and a CLI that never invokes systemd or SSH.
- [ ] Add the current pj_gpuserver2 mapping, with GPU 6 and 7 available for future instances.
- [ ] Re-run focused tests and the local CLI dry-run commands.

### Task 3: Document local usage

**Files:**
- Modify: `README.md`
- Create: `deploy/README.md`

- [ ] Document manifest fields, GPU remapping semantics, adding a Sarashina replica, and the fact that render output is local only.
- [ ] Run the complete test suite and inspect the generated plan.
