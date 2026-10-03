from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ManifestError(ValueError):
    """Raised when a GPU deployment manifest is invalid."""


_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ALLOWED_KINDS = {"asr", "qwen-tts", "kokoro", "sarashina", "cosyvoice", "ornith"}


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be an object")
    return value


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManifestError(f"{context} must be a positive integer")
    return value


def _optional_port(value: object, context: str) -> int | None:
    if value is None:
        return None
    port = _positive_int(value, context)
    if port > 65_535:
        raise ManifestError(f"{context} must be a TCP port in the range 1-65535")
    return port


@dataclass(frozen=True)
class ServiceInstance:
    id: str
    kind: str
    gpu_ids: tuple[int, ...]
    ws_port: int | None = None
    model_port: int | None = None
    enabled: bool = True
    allow_gpu_sharing: bool = False
    unit: str = ""


@dataclass(frozen=True)
class DeploymentManifest:
    version: int
    host_gpu_count: int
    services: tuple[ServiceInstance, ...]

    def validate(self) -> None:
        if self.version != 1:
            raise ManifestError(f"unsupported manifest version: {self.version}")
        if self.host_gpu_count < 1:
            raise ManifestError("host.gpu_count must be a positive integer")

        ids: set[str] = set()
        ports: dict[int, str] = {}
        gpu_users: dict[int, list[ServiceInstance]] = {}
        for service in self.services:
            if not _ID_PATTERN.fullmatch(service.id):
                raise ManifestError(
                    f"service id {service.id!r} must contain lowercase letters, digits, and hyphens"
                )
            if service.id in ids:
                raise ManifestError(f"duplicate service id: {service.id}")
            ids.add(service.id)
            if service.kind not in _ALLOWED_KINDS:
                raise ManifestError(
                    f"service {service.id!r} has unsupported kind {service.kind!r}; "
                    f"choose from {', '.join(sorted(_ALLOWED_KINDS))}"
                )
            if not service.gpu_ids:
                raise ManifestError(f"service {service.id!r} must declare at least one gpu_id")
            for gpu_id in service.gpu_ids:
                if gpu_id < 0 or gpu_id >= self.host_gpu_count:
                    raise ManifestError(
                        f"service {service.id!r} gpu_ids contains {gpu_id}; "
                        f"host has GPU IDs 0-{self.host_gpu_count - 1}"
                    )
                if service.enabled:
                    gpu_users.setdefault(gpu_id, []).append(service)

            if service.ws_port is None and service.model_port is None:
                raise ManifestError(f"service {service.id!r} needs ws_port or model_port")
            for port_name, port in (("ws_port", service.ws_port), ("model_port", service.model_port)):
                if port is None or not service.enabled:
                    continue
                previous = ports.get(port)
                if previous is not None:
                    raise ManifestError(f"port {port} is used by both {previous} and {service.id}")
                ports[port] = f"{service.id}.{port_name}"

        for gpu_id, users in gpu_users.items():
            if len(users) > 1 and not all(user.allow_gpu_sharing for user in users):
                names = ", ".join(user.id for user in users)
                raise ManifestError(
                    f"GPU {gpu_id} is shared by {names}; set allow_gpu_sharing=true on every instance"
                )

    def service(self, service_id: str) -> ServiceInstance:
        for service in self.services:
            if service.id == service_id:
                return service
        raise KeyError(service_id)

    def to_plan(self) -> dict[str, object]:
        self.validate()
        return {
            "version": self.version,
            "host": {"gpu_count": self.host_gpu_count},
            "services": [
                {
                    "id": service.id,
                    "kind": service.kind,
                    "enabled": service.enabled,
                    "gpu_ids": list(service.gpu_ids),
                    "logical_device": "cuda:0",
                    "ws_port": service.ws_port,
                    "model_port": service.model_port,
                    "allow_gpu_sharing": service.allow_gpu_sharing,
                    "unit": service.unit,
                }
                for service in self.services
            ],
        }


def load_manifest(path: Path) -> DeploymentManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc.msg}") from exc

    document = _require_mapping(raw, "manifest")
    version = document.get("version")
    if version != 1:
        raise ManifestError("manifest version must be 1")
    host = _require_mapping(document.get("host"), "host")
    gpu_count = _positive_int(host.get("gpu_count"), "host.gpu_count")
    raw_services = document.get("services")
    if not isinstance(raw_services, list) or not raw_services:
        raise ManifestError("services must be a non-empty array")

    services: list[ServiceInstance] = []
    for index, raw_service in enumerate(raw_services):
        service = _require_mapping(raw_service, f"services[{index}]")
        service_id = service.get("id")
        kind = service.get("kind")
        gpu_ids = service.get("gpu_ids")
        if not isinstance(service_id, str) or not service_id:
            raise ManifestError(f"services[{index}].id must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise ManifestError(f"services[{index}].kind must be a non-empty string")
        if not isinstance(gpu_ids, list) or not gpu_ids:
            raise ManifestError(f"services[{index}].gpu_ids must be a non-empty array")
        parsed_gpu_ids: list[int] = []
        for gpu_index, gpu_id in enumerate(gpu_ids):
            if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
                raise ManifestError(f"services[{index}].gpu_ids[{gpu_index}] must be an integer")
            if gpu_id not in parsed_gpu_ids:
                parsed_gpu_ids.append(gpu_id)
        allow_gpu_sharing = service.get("allow_gpu_sharing", False)
        if not isinstance(allow_gpu_sharing, bool):
            raise ManifestError(f"services[{index}].allow_gpu_sharing must be boolean")
        enabled = service.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ManifestError(f"services[{index}].enabled must be boolean")
        unit = service.get("unit", "")
        if not isinstance(unit, str):
            raise ManifestError(f"services[{index}].unit must be a string")
        services.append(
            ServiceInstance(
                id=service_id,
                kind=kind,
                gpu_ids=tuple(parsed_gpu_ids),
                ws_port=_optional_port(service.get("ws_port"), f"services[{index}].ws_port"),
                model_port=_optional_port(service.get("model_port"), f"services[{index}].model_port"),
                enabled=enabled,
                allow_gpu_sharing=allow_gpu_sharing,
                unit=unit,
            )
        )

    manifest = DeploymentManifest(version=version, host_gpu_count=gpu_count, services=tuple(services))
    manifest.validate()
    return manifest


def _environment_for(service: ServiceInstance) -> str:
    lines = [
        f"SERVICE_ID={service.id}",
        f"SERVICE_KIND={service.kind}",
        f"SERVICE_ENABLED={'true' if service.enabled else 'false'}",
        f"SERVICE_GPU_IDS={','.join(str(gpu_id) for gpu_id in service.gpu_ids)}",
        f"CUDA_VISIBLE_DEVICES={','.join(str(gpu_id) for gpu_id in service.gpu_ids)}",
        "MODEL_DEVICE=cuda:0",
    ]
    if service.ws_port is not None:
        lines.append(f"WS_PORT={service.ws_port}")
    if service.model_port is not None:
        lines.append(f"MODEL_PORT={service.model_port}")
    logical_device_names = {
        "asr": "ASR_DEVICE",
        "sarashina": "SARASHINA_TTS_DEVICE",
        "kokoro": "KOKORO_TTS_DEVICE",
        "cosyvoice": "COSYVOICE_DEVICE",
    }
    if device_name := logical_device_names.get(service.kind):
        lines.append(f"{device_name}=cuda:0")
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def render_manifest(manifest: DeploymentManifest, output_dir: Path) -> None:
    manifest.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        output_dir / "plan.json",
        json.dumps(manifest.to_plan(), indent=2, ensure_ascii=False) + "\n",
    )
    for service in manifest.services:
        _write_text_atomic(output_dir / f"{service.id}.env", _environment_for(service))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and render a local GPU service manifest")
    parser.add_argument("command", choices=("validate", "plan", "render"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Output directory for the render command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.config)
        if args.command == "validate":
            print(f"valid: {args.config} ({len(manifest.services)} services)")
        elif args.command == "plan":
            print(json.dumps(manifest.to_plan(), indent=2, ensure_ascii=False))
        else:
            if args.output is None:
                raise ManifestError("--output is required for render")
            render_manifest(manifest, args.output)
            print(f"rendered {len(manifest.services)} services to {args.output}")
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
