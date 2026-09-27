# Qwen3-TTS WSS 部署架构

> 状态：已部署并完成端到端冒烟验证  
> 更新日期：2026-09-23  
> 适用主机：`L20-8-hk`，远程用户 `roger`  
> 文档范围：Qwen3-TTS 模型、OpenAI 兼容 HTTP 推理服务、Qwen WSS 适配器和现有 TLS 网关

## 1. 目标与边界

本部署向业务客户端提供一个带 TLS 和 Bearer 鉴权的 Qwen3-TTS WebSocket 接口。客户端发送一条 JSON 合成请求，服务端先返回音频开始事件，再以二进制 WebSocket 帧流式返回 24 kHz、单声道、PCM S16LE 音频，最后返回完成事件。

本架构只描述 Qwen3-TTS 链路。ASR、CosyVoice、LLM、监控平台和其他模型不属于本服务的启动依赖；它们可能与本服务共享同一台主机、GPU 或网关，但应按各自的部署文档运维。

## 2. 总体架构

```mermaid
flowchart LR
    C[业务客户端]
    G[voice-api-gateway\nNginx TLS 网关\n公网 TCP 18443]
    A[sts-ws-qwen.service\nFastAPI/Uvicorn WSS 适配器\n172.19.0.1:18100]
    V[qwen3-tts-vllm.service\nvLLM-Omni OpenAI 兼容 API\n172.19.0.1:8091]
    GPU[GPU 0\nNVIDIA L20]
    M[/app/sts_ws_server/models/tts/qwen3\nQwen3-TTS-12Hz-1.7B-CustomVoice]

    C -->|WSS /tts/v1/realtime\nBearer token| G
    G -->|WebSocket Upgrade\nAuthorization 原样转发| A
    A -->|HTTP POST /v1/audio/speech\nPCM streaming| V
    V -->|模型推理| GPU
    M -->|本地权重| V
    V -->|PCM HTTP 流| A
    A -->|事件 + PCM 二进制帧| G
    G --> C
```

数据流按以下顺序执行：

1. 客户端连接 `wss://47.87.138.166:18443/tts/v1/realtime`，携带 `Authorization: Bearer <token>`。
2. 现有 `voice-api-gateway` 完成公网 TLS、调用方校验、限流和 WebSocket Upgrade。
3. Nginx 将请求转发到 `http://172.19.0.1:18100`，保留原始 `Authorization` 头、`X-Caller` 和 `X-Request-ID`。
4. WSS 适配器校验 Bearer token，将 JSON 请求转换为 `POST /v1/audio/speech`。
5. vLLM-Omni 从本地模型目录读取权重，在 GPU 0 上生成 PCM 流。
6. 适配器按偶数字节边界转发 PCM，客户端收到 `response.completed` 后结束本次请求。

## 3. 组件职责与部署位置

| 组件 | 运行位置 | 当前监听 | 主要职责 |
| --- | --- | --- | --- |
| 业务客户端 | 业务方服务器或应用 | 出站 WSS | 建立连接、发送文本、播放 PCM |
| `voice-api-gateway` | Docker 容器 | 公网 `18443` 映射到容器 `443` | TLS 终止、Bearer 调用方校验、限流、WebSocket 代理 |
| `sts-ws-qwen.service` | 主机 systemd 系统服务 | `172.19.0.1:18100` | WSS 协议、输入校验、鉴权、HTTP 请求适配、PCM 转发 |
| `qwen3-tts-vllm.service` | 主机 systemd 系统服务 | `172.19.0.1:8091` | 加载 Qwen3-TTS、提供 `/v1/audio/speech` |
| Qwen3-TTS 权重 | 主机文件系统 | `/app/sts_ws_server/models/tts/qwen3` | 模型配置、tokenizer、主模型和 speech tokenizer |
| 适配器 Python 环境 | 主机 venv | `/app/sts_ws_server/venv` | FastAPI、Uvicorn、HTTP 客户端和 WSS 代码 |
| 共享 Python 环境 | 主机 venv | `/app/sts_ws_server/venv` | Python 3.13、FastAPI、vLLM、vLLM-Omni、CUDA PyTorch |

两个 systemd 服务都以 `roger:roger` 运行，不使用 root 权限。适配器通过 `Requires=qwen3-tts-vllm.service` 和 `After=qwen3-tts-vllm.service` 声明依赖及启动顺序；两个 unit 各自使用 `Restart=on-failure`，模型服务重启后应按下文的健康检查确认适配器也已恢复。

## 4. 网络、端口与信任边界

| 端口或地址 | 协议 | 访问范围 | 说明 |
| --- | --- | --- | --- |
| `47.87.138.166:18443` | HTTPS/WSS | 公网入口 | 业务客户端唯一正式入口 |
| `172.19.0.1:18100` | HTTP/WebSocket | 主机与 `voice-internal` Docker 网络 | WSS 适配器；不应开放公网 |
| `172.19.0.1:8091` | HTTP | 主机与内部 Docker 网络 | vLLM OpenAI 兼容 API；不应开放公网 |
| `127.0.0.1`/本地 Unix 权限 | 文件 | 主机 | `wss.env`、模型和 systemd unit |

网关配置使用精确路由 `location = /tts/v1/realtime`，上游为 `http://172.19.0.1:18100`。网关当前设置如下：

- `proxy_http_version 1.1`，并转发 `Upgrade`/`Connection`；
- 关闭 request/response buffering，允许音频边生成边发送；
- 连接超时 10 秒，读写超时 3600 秒；
- `limit_req` 使用 `burst=20 nodelay`；
- `limit_conn` 当前上限为 20 条活动连接。

安全组和主机防火墙只应允许业务方出口访问 TCP 18443。18100 和 8091 即使绑定在 Docker bridge 主机地址，也不得作为公网 API 使用。

## 5. 模型文件与运行时

模型来源为 [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)。vLLM 以本地目录启动，服务端设置 `HF_HUB_OFFLINE=1`，运行时不依赖再次访问 Hugging Face。

当前模型目录已经是非空且完整的本地快照：

```text
/app/sts_ws_server/models/tts/qwen3/
├── config.json
├── generation_config.json
├── merges.txt
├── model.safetensors                         3,833,402,552 bytes
├── preprocessor_config.json
├── tokenizer_config.json
├── vocab.json
└── speech_tokenizer/
    ├── config.json
    ├── configuration.json
    ├── model.safetensors                       682,293,092 bytes
    └── preprocessor_config.json
```

部署核对时发现，之前目录根部显示的 `model.safetensors`、大小为 `682,293,092` 字节的文件对应仓库中的 `speech_tokenizer/model.safetensors`。它已保留在正确的 `speech_tokenizer/` 子目录；根目录的主模型是另一个约 3.83 GB 的 `model.safetensors`。模型目录非空时不得用空目录初始化逻辑覆盖它，也不得在没有校验和或文件大小记录的情况下删除旧权重。

当前关键运行时版本：

| 项目 | 当前值 |
| --- | --- |
| vLLM | `0.24.0` |
| vLLM-Omni | `0.24.0` |
| PyTorch | `2.11.0+cu130` |
| CUDA runtime | `13.0` |
| 共享 Python | 3.13.15 环境 `/app/sts_ws_server/venv` |
| 旧回滚环境 | `/app/sts_ws_server/venv-py314-old`（仅保留回滚） |
| 模型别名 | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| 推理 GPU | `CUDA_VISIBLE_DEVICES=0`，NVIDIA L20 |
| 最大模型长度 | vLLM `/v1/models` 当前报告 `4096` |

vLLM 的 `--enforce-eager` 会降低启动期间的编译和显存峰值不确定性，但可能牺牲部分性能；除非重新做显存和延迟验收，不要在升级时随意移除。

## 6. 服务启动配置

### 6.1 vLLM-Omni 模型服务

对应 unit：`/etc/systemd/system/qwen3-tts-vllm.service`。

```text
WorkingDirectory=/app/sts_ws_server
User=roger
Environment=CUDA_VISIBLE_DEVICES=0
Environment=HF_HUB_OFFLINE=1
Environment=PYTHONUNBUFFERED=1
Environment=PATH=/app/sts_ws_server/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
ExecStart=/app/sts_ws_server/venv/bin/vllm serve \
  /app/sts_ws_server/models/tts/qwen3 \
  --deploy-config /app/sts_ws_server/venv/lib/python3.13/site-packages/vllm_omni/deploy/qwen3_tts.yaml \
  --omni --host 172.19.0.1 --port 8091 \
  --trust-remote-code --enforce-eager \
  --served-model-name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
Restart=on-failure
RestartSec=15
TimeoutStartSec=0
LimitNOFILE=1048576
```

`PATH` 显式包含共享 venv 的 `bin`，因为 vLLM-Omni 初始化可能调用该环境中的 `ninja`。systemd 环境不能假设交互式 shell 的 PATH 已经生效。

### 6.2 WSS 适配器

对应 unit：`/etc/systemd/system/sts-ws-qwen.service`。

```text
WorkingDirectory=/app/sts_ws_server/service-src
EnvironmentFile=/app/sts_ws_server/wss.env
Environment=SERVICE_NAME=qwen-tts
Environment=QWEN_TTS_BACKEND=openai
Environment=QWEN_TTS_BASE_URL=http://172.19.0.1:8091/v1
Environment=QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
Environment=QWEN_TTS_VOICE=aiden
Environment=MAX_CONCURRENT_REQUESTS=1
Environment=MODEL_TIMEOUT_SECONDS=300
ExecStart=/app/sts_ws_server/venv/bin/uvicorn \
  sts_ws_server.app:app --host 172.19.0.1 --port 18100 --workers 1
Restart=on-failure
RestartSec=5
LimitNOFILE=65535
```

`wss.env` 只保存 Bearer token 等敏感配置，权限应为 `0600`，属主为 `roger:roger`。本文档不记录任何 token 值。适配器默认 `MAX_TEXT_CHARS=10000`、`TTS_AUDIO_CHUNK_BYTES=4096`、`QWEN_TTS_SAMPLE_RATE=24000`。

## 7. WSS 与 HTTP 协议

### 7.1 建立连接

```text
wss://47.87.138.166:18443/tts/v1/realtime
Authorization: Bearer <已获发的业务 token>
```

当前 TLS 证书为自签名证书，SAN 包含 `47.87.138.166`。生产客户端应通过受控渠道取得服务器上的 `mimo-model-ip.crt` 并配置 CA 校验；测试时可以临时关闭校验，但不能作为生产配置。

### 7.2 客户端请求

连接成功后发送 JSON 文本帧：

```json
{
  "type": "response.create",
  "text": "こんにちは。ミモの音声サービスです。",
  "voice": "Ono_Anna",
  "language": "ja"
}
```

字段规则：

- `type` 必须是 `response.create`；
- `text` 必填且必须是非空字符串，默认最多 10,000 个字符；
- `voice` 可选，缺省使用服务配置的 `aiden`；
- `language` 可选，会原样传给兼容 HTTP backend；
- 一个连接可以连续提交多条请求，但服务端按收到顺序处理，不提供同一连接内的响应 ID 路由。

### 7.3 服务端响应

开始生成时发送：

```json
{
  "type": "response.audio.started",
  "format": "pcm_s16le",
  "sample_rate": 24000,
  "channels": 1
}
```

随后发送多个二进制帧。每个帧都是 little-endian、16-bit、mono、24 kHz PCM，不包含 WAV header，不使用 base64。模型 HTTP 流的块大小默认 4096 字节；适配器会处理跨 HTTP chunk 的半个 PCM sample，确保不会输出奇数字节帧。全部音频发送完成后发送：

```json
{"type":"response.completed"}
```

输入格式错误、鉴权失败、上游 HTTP 错误、超时或没有音频时，服务端返回：

```json
{"type":"error","message":"..."}
```

未通过鉴权的 WebSocket 在接受前关闭，关闭码为 `4401`；未启用的路径关闭码为 `4404`。

### 7.4 适配器到模型 API

适配器调用：

```http
POST http://172.19.0.1:8091/v1/audio/speech
Content-Type: application/json
Authorization: Bearer <若配置 QWEN_TTS_API_KEY>
```

请求体的核心字段为：

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "input": "こんにちは。",
  "voice": "Ono_Anna",
  "response_format": "pcm",
  "stream_format": "audio",
  "language": "ja"
}
```

上游必须返回可流式读取的原始 PCM。适配器不负责把 WAV、MP3 或其他压缩格式转换成 PCM；若模型服务返回的不是 PCM，应在模型服务侧或单独的内部转换层解决。

## 8. 远程部署与更新流程

所有对远程服务器的操作使用仓库配套的 `/Users/rgwei/pj/pj_agent/pj_gpusever1/remote.py`，不要绕过它直接执行 `ssh` 或 `scp`。连接配置使用用户指定的密钥 `/Users/rgwei/pj/pj_agent/pj_gpusever1/id_ed25519`；私钥内容不得复制到仓库、文档或日志。

### 8.1 部署前检查

```bash
REMOTE=/Users/rgwei/pj/pj_agent/pj_gpusever1/remote.py

python3 "$REMOTE" --capture \
  "hostname; whoami; sudo -n systemctl is-active qwen3-tts-vllm.service sts-ws-qwen.service"

python3 "$REMOTE" --capture \
  "sudo -n find /app/sts_ws_server/models/tts/qwen3 -maxdepth 2 -type f -printf '%P %s bytes\\n' | sort"

python3 "$REMOTE" --capture \
  "sudo -n df -h /app; sudo -n nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv"
```

确认模型目录已存在且文件大小与登记值一致后再执行更新。当前根分区约 1,008 GB，总使用率约 96%，可用空间约 43 GB；模型替换、临时下载和日志增长都必须先评估空间。

### 8.2 部署或更新模型

1. 从 Hugging Face 下载到临时目录，不直接覆盖正在使用的目录。
2. 校验主模型、speech tokenizer、配置和 tokenizer 文件；至少记录大小及 SHA-256。
3. 确保最终目录同时包含根目录主模型和 `speech_tokenizer/model.safetensors`，不要把 682,293,092 字节的 tokenizer 文件误当成主模型。
4. 停止适配器，停止 vLLM，完成文件同步后再启动；磁盘足够时保留上一版目录用于回滚。
5. 使用 `systemctl daemon-reload`（unit 有变化时），启动 vLLM 并等待 `/v1/models` 出现预期模型名。
6. 启动 WSS 适配器，检查 `/healthz` 后再 reload 网关。

示例启动顺序：

```bash
python3 "$REMOTE" --capture \
  "sudo -n systemctl daemon-reload; sudo -n systemctl enable qwen3-tts-vllm.service sts-ws-qwen.service; sudo -n systemctl restart qwen3-tts-vllm.service"

# vLLM 首次加载可能需要较长时间；确认 API 已就绪后再重启适配器。
python3 "$REMOTE" --capture \
  "curl -fsS --max-time 10 http://172.19.0.1:8091/v1/models"

python3 "$REMOTE" --capture \
  "sudo -n systemctl restart sts-ws-qwen.service; curl -fsS --max-time 5 http://172.19.0.1:18100/healthz"
```

网关模板变更前先备份 `/app/sts_ws_server/deploy/backups/` 中的现有版本。使用现有 voice-stack 的发布流程更新 `default.conf.template`，在容器内执行 `nginx -t` 通过后 reload；不要新建第二个公网 TLS 网关。

### 8.3 回滚

若新权重不能加载或 WSS 冒烟失败：

1. 停止 `sts-ws-qwen.service` 和 `qwen3-tts-vllm.service`。
2. 将已验收的上一版模型目录恢复到固定路径，保持目录结构完整。
3. 启动 vLLM，确认模型列表和日志无加载错误。
4. 启动适配器，检查健康接口和直连 WSS。
5. 必要时恢复上一版网关模板并 reload。

不要用 `rm -rf` 清理唯一模型副本；先确认备份、校验和和可用磁盘。

## 9. 健康检查与验收

健康检查分三层：

```bash
# 1. 模型 API 和模型名
curl -fsS http://172.19.0.1:8091/v1/models

# 2. WSS 适配器进程和配置
curl -fsS http://172.19.0.1:18100/healthz

# 3. 公网 TLS 网关
curl -kfsS https://47.87.138.166:18443/healthz
```

适配器 `/healthz` 只确认进程、服务配置和 token 已加载，不会主动执行一次模型合成；模型 API 和端到端 WSS 必须单独检查。

项目内提供客户端冒烟脚本：

```bash
cd /Users/rgwei/pj/pj_audio/sts_ws_server/demo1
python3 -m pip install websocket-client
export MIMO_WSS_BASE_URL='wss://47.87.138.166:18443'
export MIMO_API_KEY='<已获发的业务 token>'
export MIMO_CA_CERT='/path/to/mimo-model-ip.crt'
python3 test_tts_wss.py --text 'こんにちは。ミモの音声サービスです。'
```

脚本应观察到 `response.audio.started`、至少一个二进制音频帧和 `response.completed`，并生成可播放的 `tts-wss-test.wav`。当前已完成的验收记录：模型 HTTP 合成返回 200；直连适配器 WSS 返回 PCM；经 `18443` 网关的端到端 WSS 返回 PCM16，两个 systemd 服务为 `enabled` 且 `active`，Nginx 配置测试通过。

## 10. 日志、监控与故障排查

### 10.1 常用日志命令

```bash
python3 "$REMOTE" --capture \
  "sudo -n journalctl -u qwen3-tts-vllm.service -n 200 --no-pager"

python3 "$REMOTE" --capture \
  "sudo -n journalctl -u sts-ws-qwen.service -n 200 --no-pager"

python3 "$REMOTE" --capture \
  "sudo -n docker logs --tail 200 voice-api-gateway"
```

请求正文、Bearer token 和模型输出不应写入应用日志。排障时可以记录 request ID、状态码、延迟、音频字节数和错误类型。

### 10.2 故障矩阵

| 现象 | 优先检查 | 处理 |
| --- | --- | --- |
| 公网返回 `401` | Bearer token、网关 `$caller` 规则、证书入口 | 确认客户端使用已获发 token；不要把内部文件内容发给客户端 |
| 公网返回 `404` | 路径是否精确为 `/tts/v1/realtime`、模板是否已 reload | 检查网关模板和容器内 `nginx -t` |
| 公网返回 `502` | `sts-ws-qwen.service` 状态、18100 监听、Docker bridge 路由 | 先查适配器日志，再查 8091 模型 API |
| 适配器 `/healthz` 为 `not_ready` | `wss.env` 权限、`WSS_API_KEY`、`SERVICE_NAME` | 修正环境文件后重启适配器 |
| vLLM 反复重启 | `journalctl` 中模型加载、CUDA、`ninja` 错误 | 确认模型结构、磁盘、PATH 和 GPU 显存 |
| `/v1/models` 无预期模型 | `--served-model-name`、模型目录、配置文件 | 检查 root `model.safetensors` 与 `speech_tokenizer/` 是否完整 |
| WSS 有 started 但无音频 | 上游 `/v1/audio/speech` 状态、返回 Content-Type/字节数 | 确认上游返回 raw PCM，而不是 WAV/MP3 或空响应 |
| 音频无法播放 | `sample_rate=24000`、PCM S16LE、mono、偶数字节 | 检查客户端按事件声明创建音频流，不要当作 WAV 直接写入带 header 的文件 |
| 延迟或超时 | GPU 0 竞争、vLLM 日志、并发限制、网关 3600 秒超时 | 保持单 worker，评估显存与 `MAX_CONCURRENT_REQUESTS` 后再调整 |
| 磁盘告警 | `df -h /app`、模型下载缓存、journal 大小 | 先删除可重建的临时缓存，保留当前模型和可回滚副本 |

曾遇到的启动问题是 systemd 的 PATH 不包含 vLLM 环境中的 `ninja`；当前 unit 已显式设置 PATH。后续编辑 unit 时不得删除该设置。

## 11. 容量与运行限制

- 适配器 `MAX_CONCURRENT_REQUESTS=1`，同一进程只允许一条模型推理进入；不要通过增加 Uvicorn workers 复制模型。
- 单请求文本上限默认 10,000 字符；模型服务当前报告最大上下文长度 4096，超长文本可能导致上游拒绝或显著增加延迟。
- 上游调用超时为 300 秒，网关 WebSocket 读写超时为 3600 秒。
- `TTS_AUDIO_CHUNK_BYTES=4096` 仅控制 HTTP 读取和转发块大小，不代表单个语音句子的最大长度。
- vLLM 当前只使用 GPU 0；主机其他 GPU 上有其他工作负载时，应将它们视作共享资源，不要据此假设整机空闲。
- 根分区当前约 96% 已用、可用约 43 GB。模型副本、下载缓存、日志和临时文件应纳入容量告警；建议在低于 20 GB 可用空间前停止模型升级操作。
- 业务方应通过网关限流控制连接数；`limit_conn=20` 是连接上限，不等于允许 20 路并行 GPU 推理。

## 12. 配置与变更清单

| 配置项 | 当前值或位置 | 变更注意 |
| --- | --- | --- |
| 模型目录 | `/app/sts_ws_server/models/tts/qwen3` | 非空目录；更新前做校验和和备份 |
| vLLM unit | `/etc/systemd/system/qwen3-tts-vllm.service` | GPU、端口、PATH、模型别名变更需重新验收 |
| WSS unit | `/etc/systemd/system/sts-ws-qwen.service` | 单 worker、18100、上游 URL 和并发限制 |
| WSS 环境 | `/app/sts_ws_server/wss.env` | 权限 `0600`；不得提交或打印 token |
| WSS 源码 | `/app/sts_ws_server/service-src` | 与本地 `sts_ws_server` 版本保持可追溯 |
| 网关模板 | `/home/scarlett/mimo_services/voice-stack/gateway/templates/default.conf.template` | 只改精确 TTS 路由；变更前备份和 `nginx -t` |
| 网关证书 | 现有 voice-stack TLS 目录 | 自签名证书客户端需配置 CA；证书更新需重新验证 SAN |
| 远程连接 | `/Users/rgwei/pj/pj_agent/pj_gpusever1/remote.py` | 使用指定 `id_ed25519`；不直接使用裸 ssh/scp |

## 13. 发布完成标准

一次 TTS 发布只有在以下条件全部满足时才算完成：

- 模型目录包含主模型、speech tokenizer、配置和 tokenizer，大小或校验和与发布记录一致；
- `qwen3-tts-vllm.service` 和 `sts-ws-qwen.service` 均为 `enabled`、`active`；
- `/v1/models` 返回预期模型名，适配器 `/healthz` 返回 `status=ok`；
- 网关模板中的精确路由仍指向 `172.19.0.1:18100`，`nginx -t` 通过；
- 缺失或错误 Bearer token 不可建立连接；
- 直连适配器和公网 `18443` 入口均能收到 started、PCM 二进制帧和 completed；
- 客户端用受信任 CA 校验自签名证书，生成的 PCM 可按 24 kHz mono PCM16 播放；
- 发布后重新检查磁盘、GPU 显存、systemd 日志和网关错误率；
- 记录模型版本、部署时间、回滚目录和验收结果，但不记录任何 token 或私钥内容。
