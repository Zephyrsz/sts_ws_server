# GPU deployment manifest

`gpu-services.json` is the local source of truth for the GPU and port assignment of each service group. It describes the current `pj_gpuserver2` layout and keeps disabled replica slots for future expansion.

The manifest is intentionally JSON so the renderer runs with the Python standard library and does not need to install a YAML package on the model host. It does not contain API keys or other secrets; those remain in the existing `wss.env`, `sarashina.env`, and ASR environment files.

## Validate and render locally

Run these commands from the repository root:

```bash
python3 deploy/gpu_deployment.py validate \
  --config deploy/gpu-services.json

python3 deploy/gpu_deployment.py plan \
  --config deploy/gpu-services.json

python3 deploy/gpu_deployment.py render \
  --config deploy/gpu-services.json \
  --output /tmp/sts-gpu-rendered
```

`render` writes a normalized `plan.json` and one safe environment fragment per service. It does not call SSH, `systemctl`, or Nginx, and it does not change the remote host. Generated output should be reviewed before a later deployment step.

## GPU numbering

`gpu_ids` are physical host IDs. The generated `CUDA_VISIBLE_DEVICES` preserves those IDs, while `MODEL_DEVICE=cuda:0` and the service-specific device variables refer to the first logical device exposed to that process. For example, a Sarashina entry with `gpu_ids: [6]` runs with `CUDA_VISIBLE_DEVICES=6` and `SARASHINA_TTS_DEVICE=cuda:0`.

Multi-GPU models list all physical IDs, as `ornith-0` does with `[1, 2]`. A service that intentionally shares a GPU must set `allow_gpu_sharing: true`; validation rejects accidental sharing.

## Adding a Sarashina replica

Copy the `sarashina-1` entry, choose a new ID, an unused GPU, and an unused WebSocket port, then set `enabled` to `true`:

```json
{
  "id": "sarashina-1",
  "kind": "sarashina",
  "gpu_ids": [6],
  "ws_port": 18103,
  "enabled": true,
  "unit": "sarashina-ws@1.service"
}
```

The current file leaves GPU 7 available for another replica or another service. Enabling a replica only changes the local manifest until a separate, explicitly reviewed deployment step consumes the rendered plan.
