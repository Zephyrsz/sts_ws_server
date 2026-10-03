import json
from pathlib import Path

import pytest

from deploy.gpu_deployment import ManifestError, load_manifest, render_manifest


MANIFEST = Path(__file__).parents[1] / "deploy" / "gpu-services.json"


def test_current_gpu_manifest_matches_pj_gpuserver2_snapshot() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.host_gpu_count == 8
    assert manifest.service("qwen-tts-0").gpu_ids == (0,)
    assert manifest.service("ornith-0").gpu_ids == (1, 2)
    assert manifest.service("sarashina-0").gpu_ids == (3,)
    assert manifest.service("asr-0").gpu_ids == (4,)
    assert manifest.service("cosyvoice-0").gpu_ids == (5,)
    assert manifest.service("sarashina-1").enabled is False


def test_multiple_sarashina_instances_are_distinct() -> None:
    manifest = load_manifest(MANIFEST)
    sarashina = [service for service in manifest.services if service.kind == "sarashina"]

    assert [service.id for service in sarashina] == ["sarashina-0", "sarashina-1"]
    assert sarashina[0].ws_port == 18102
    assert sarashina[1].ws_port == 18103


def test_rejects_gpu_outside_host_capacity(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "version": 1,
            "host": {"gpu_count": 2},
            "services": [{"id": "asr-0", "kind": "asr", "gpu_ids": [2], "ws_port": 8011}],
        },
    )

    with pytest.raises(ManifestError, match="gpu_ids.*2"):
        load_manifest(path)


def test_rejects_duplicate_ports(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "version": 1,
            "host": {"gpu_count": 2},
            "services": [
                {"id": "asr-0", "kind": "asr", "gpu_ids": [0], "ws_port": 8011},
                {"id": "asr-1", "kind": "asr", "gpu_ids": [1], "ws_port": 8011},
            ],
        },
    )

    with pytest.raises(ManifestError, match="port 8011"):
        load_manifest(path)


def test_rejects_port_outside_tcp_range(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "version": 1,
            "host": {"gpu_count": 1},
            "services": [{"id": "asr-0", "kind": "asr", "gpu_ids": [0], "ws_port": 65536}],
        },
    )

    with pytest.raises(ManifestError, match="ws_port must be a TCP port"):
        load_manifest(path)


def test_rejects_unapproved_gpu_sharing(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        {
            "version": 1,
            "host": {"gpu_count": 2},
            "services": [
                {"id": "asr-0", "kind": "asr", "gpu_ids": [0], "ws_port": 8011},
                {"id": "asr-1", "kind": "asr", "gpu_ids": [0], "ws_port": 8012},
            ],
        },
    )

    with pytest.raises(ManifestError, match="GPU 0 is shared"):
        load_manifest(path)


def test_render_writes_plan_and_safe_environment_files(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)

    render_manifest(manifest, tmp_path / "rendered")

    plan = json.loads((tmp_path / "rendered" / "plan.json").read_text())
    assert plan["services"][0]["id"] == "qwen-tts-0"
    assert plan["services"][-1]["enabled"] is False

    sarashina_env = (tmp_path / "rendered" / "sarashina-0.env").read_text()
    assert "CUDA_VISIBLE_DEVICES=3" in sarashina_env
    assert "SERVICE_GPU_IDS=3" in sarashina_env
    assert "SARASHINA_TTS_DEVICE=cuda:0" in sarashina_env
    assert "WSS_API_KEY" not in sarashina_env


def _write_manifest(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path
