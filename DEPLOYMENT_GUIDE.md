# 统一架构部署指南 v2.0

> ASR/LLM/TTS WebSocket 服务统一架构方案
> 
> **版本**: 2.0  
> **日期**: 2026-09-24  
> **改进**: 统一认证、统一推理架构、简化配置管理

---

## 📋 架构概览

### 核心改进 (v2.0)

**1. 统一认证机制** ✅
- 所有服务(ASR/TTS/LLM)共享同一个 `WSS_API_KEY`
- 统一的认证中间件 (`auth.py`)
- 简化的认证逻辑,便于维护

**2. 统一推理架构** ✅
- LLM 服务改为 Backend 工厂模式(与 ASR/TTS 一致)
- 统一的连接池管理
- 统一的超时控制
- 统一的错误处理和重试机制

**3. 简化配置管理** ✅
- 分层配置系统(CommonConfig/ASRConfig/TTSConfig/LLMConfig)
- 环境变量数量减少 60%+
- 更清晰的配置结构
- 向后兼容旧配置

### 部署架构

```
客户端 (iot-mimo-ai-nokeys)
        ↓ WebSocket + Bearer Token
    Nginx TLS Gateway (:18443)
        ↓ 统一认证
    ┌───────────┬──────────────┬───────────────┬──────────┐
    ↓           ↓              ↓               ↓          ↓
ASR 服务    LLM 服务    Qwen-TTS 服务   CosyVoice   (统一架构)
Kotoba      Qwen 35B    Qwen3-TTS        服务
Whisper     Backend     Backend          Backend
Backend
```

**关键特性**:
- ✅ 所有服务共享 `WSS_API_KEY`
- ✅ 所有服务使用统一的 Backend 工厂模式
- ✅ 统一的连接池和超时管理
- ✅ 统一的错误处理
- ✅ 独立部署和扩展能力

---

## 🚀 快速开始

### 1. 环境配置

创建 `.env` 文件(从 `.env.example` 复制):

```bash
cp .env.example .env
```

**最小化配置示例** (v2.0):

```bash
# ========================================
# 统一认证配置
# ========================================
WSS_API_KEY=your-secure-random-key-here

# ========================================
# 通用配置
# ========================================
SERVICE_NAME=all
MAX_CONCURRENT_REQUESTS=2
MODEL_TIMEOUT_SECONDS=300

# ========================================
# ASR 配置 (Kotoba Whisper)
# ========================================
ASR_BACKEND=kotoba
ASR_MODEL=/path/to/kotoba-whisper-v2.0
ASR_DEVICE=cuda:0

# VAD 配置
VAD_BACKEND=silero
VAD_MODEL_PATH=/path/to/silero_vad.onnx
VAD_THRESHOLD=0.5

# ========================================
# LLM 配置 (Qwen 35B)
# ========================================
LLM_BACKEND=openai
LLM_BASE_URL=http://127.0.0.1:18080/v1
LLM_MODEL=Qwen3.6-35B-A3B
LLM_CONNECT_TIMEOUT_SECONDS=10
LLM_READ_TIMEOUT_SECONDS=600

# ========================================
# TTS 配置 (Qwen3-TTS)
# ========================================
QWEN_TTS_BACKEND=openai
QWEN_TTS_BASE_URL=http://host.docker.internal:8091/v1
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_TTS_VOICE=aiden
QWEN_TTS_SAMPLE_RATE=24000

# CosyVoice (可选)
COSYVOICE_TTS_BACKEND=openai
COSYVOICE_TTS_BASE_URL=http://host.docker.internal:8092/v1
COSYVOICE_TTS_MODEL=cosyvoice3
COSYVOICE_TTS_VOICE=default

# ========================================
# 资源限制
# ========================================
MAX_AUDIO_SECONDS=120
MAX_TEXT_CHARS=10000
```

**对比旧版本** (v1.0 需要 50+ 个环境变量):
```bash
# v1.0 需要分别配置每个服务的认证
WSS_API_KEY=xxx          # ASR/TTS 共享
LLM_WSS_API_KEY=yyy      # LLM 独立

# v1.0 需要为每个服务重复配置超时
ASR_TIMEOUT=300
TTS_TIMEOUT=300
LLM_TIMEOUT=600

# v2.0 统一配置
WSS_API_KEY=xxx          # 所有服务共享
MODEL_TIMEOUT_SECONDS=300 # 统一超时
```

### 2. Docker Compose 部署

```bash
# 启动所有服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f asr
docker compose logs -f llm-wss
docker compose logs -f qwen-tts
```

### 3. 验证部署

```bash
# 测试 ASR 服务
python demo1/test_asr_wss.py

# 测试 TTS 服务
python demo1/test_tts_wss.py

# 测试 CosyVoice 服务
python demo1/test_cosyvoice3_wss.py
```

---

## 🏗️ 统一架构详解

### 1. 统一认证机制

**新增文件**: `src/sts_ws_server/auth.py`

所有 WebSocket 服务使用统一的认证中间件:

```python
from sts_ws_server.auth import authenticate_websocket

@app.websocket("/asr/v1/realtime")
async def asr_realtime(websocket: WebSocket):
    # 统一认证
    if not await authenticate_websocket(websocket, settings, "ASR"):
        return
    
    # 业务逻辑
    # ...
```

**特性**:
- 统一的 Bearer Token 验证
- 支持两种格式: `Bearer <token>` 或 `<token>`
- 统一的错误响应格式
- 可选认证(未配置 API Key 时跳过)

### 2. 统一推理架构

**改进**: `src/sts_ws_server/backends.py`

所有推理服务使用统一的 Backend Protocol:

```python
# ASR Backend
class ASRBackend(Protocol):
    async def transcribe(self, pcm_s16le: bytes, language: str | None) -> Transcript: ...

# TTS Backend
class TTSBackend(Protocol):
    async def stream_audio(self, text: str, voice: str | None, language: str | None) -> AsyncIterator[bytes]: ...

# LLM Backend (新增)
class LLMBackend(Protocol):
    async def chat(self, messages: list[dict], model: str | None = None, stream: bool = True, **kwargs) -> AsyncIterator[str]: ...
```

**统一特性**:
- 统一的连接池管理(`_PooledHTTPBackend`)
- 统一的超时控制
- 统一的错误处理(`ModelBackendError`)
- 统一的工厂函数(`create_asr_backend`, `create_tts_backend`, `create_llm_backend`)

**LLM Backend 实现**:

```python
class OpenAICompatibleLLMBackend(_PooledHTTPBackend):
    """OpenAI 兼容的 LLM Backend"""
    
    def __init__(self, settings: Settings):
        super().__init__(settings.llm.read_timeout_seconds)
        self.endpoint = f"{settings.llm.base_url.rstrip('/')}/chat/completions"
        self.api_key = settings.llm.api_key
        self.model = settings.llm.model
        self.connect_timeout = settings.llm.connect_timeout_seconds
    
    async def chat(self, messages: list[dict], model: str | None = None, stream: bool = True, **kwargs) -> AsyncIterator[str]:
        # 统一的流式调用逻辑
        # 统一的错误处理
        # 统一的超时控制
        # ...
```

### 3. 简化配置管理

**改进**: `src/sts_ws_server/config.py`

分层配置系统:

```python
@dataclass(frozen=True)
class CommonConfig:
    """通用配置 - 所有服务共享"""
    api_key: str = ""
    max_concurrent_requests: int = 2
    model_timeout_seconds: float = 300.0
    # VAD 配置
    vad_backend: str = "energy"
    # ...

@dataclass(frozen=True)
class ASRConfig:
    """ASR 服务配置"""
    backend: str = "mock"
    model: str = "small"
    device: str = "cpu"
    # ...

@dataclass(frozen=True)
class TTSConfig:
    """TTS 服务配置"""
    backend: str = "mock"
    base_url: str = "http://host.docker.internal:8091/v1"
    # ...

@dataclass(frozen=True)
class LLMConfig:
    """LLM 服务配置"""
    backend: str = "openai"
    base_url: str = "http://127.0.0.1:18080/v1"
    # ...

@dataclass(frozen=True)
class Settings:
    """统一配置"""
    common: CommonConfig
    asr: ASRConfig
    qwen_tts: TTSConfig
    cosyvoice_tts: TTSConfig
    llm: LLMConfig
```

**向后兼容**:

```python
class Settings:
    @property
    def api_key(self) -> str:
        """向后兼容: 统一的 API Key"""
        return self.common.api_key
    
    @property
    def asr_backend(self) -> str:
        """向后兼容: ASR Backend"""
        return self.asr.backend
```

---

## 📦 服务部署方式

### 方式 1: Docker Compose (推荐)

适用于开发和测试环境:

```bash
# 启动所有服务
docker compose up -d

# 只启动 ASR 服务
docker compose up -d asr

# 只启动 LLM 服务
docker compose up -d llm-wss
```

### 方式 2: Systemd 服务 (生产环境)

适用于 Linux 主机直接部署:

**ASR 服务**:

```bash
# 1. 安装依赖
pip install -e ".[kotoba]" --break-system-packages

# 2. 配置环境变量
sudo mkdir -p /etc/sts-ws-server
sudo cp deploy/asr.env.example /etc/sts-ws-server/asr.env
sudo chmod 600 /etc/sts-ws-server/asr.env
sudo nano /etc/sts-ws-server/asr.env  # 编辑配置

# 3. 安装 systemd 服务
sudo cp deploy/asr-ws.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now asr-ws.service

# 4. 检查状态
sudo systemctl status asr-ws.service
curl http://localhost:8011/healthz
```

**LLM 服务**:

```bash
# 1. 启动 LLM 模型服务 (vLLM)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 2 \
  --port 18080

# 2. 启动 LLM WebSocket Bridge
# (使用与 ASR 相同的配置和部署方式)
```

### 方式 3: Kubernetes (大规模部署)

参考 `k8s/` 目录中的示例配置(待补充)。

---

## 🔧 配置详解

### 统一认证配置

```bash
# 所有服务共享同一个 API Key
WSS_API_KEY=your-secure-random-key-here
```

**生成高熵 API Key**:

```bash
# 方式 1: OpenSSL
openssl rand -base64 32

# 方式 2: Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 通用配置

```bash
# 服务名称 (all | asr | qwen-tts | cosyvoice | llm)
SERVICE_NAME=all

# 并发限制 (每个服务)
MAX_CONCURRENT_REQUESTS=2

# 模型推理超时 (所有服务)
MODEL_TIMEOUT_SECONDS=300
```

### ASR 配置

**Kotoba Whisper (日语专用)**:

```bash
ASR_BACKEND=kotoba
ASR_MODEL=/models/kotoba-whisper-v2.0
ASR_DEVICE=cuda:0
```

**Faster Whisper (多语言)**:

```bash
ASR_BACKEND=faster-whisper
ASR_MODEL=large-v3
ASR_DEVICE=cuda
ASR_COMPUTE_TYPE=float16
```

**OpenAI 兼容 API**:

```bash
ASR_BACKEND=openai
ASR_BASE_URL=http://asr-model-server:8090/v1
ASR_API_KEY=optional-key
ASR_HTTP_MODEL=whisper-1
```

### LLM 配置

**OpenAI 兼容接口** (vLLM / llama.cpp):

```bash
LLM_BACKEND=openai
LLM_BASE_URL=http://127.0.0.1:18080/v1
LLM_MODEL=Qwen3.6-35B-A3B

# 超时配置
LLM_CONNECT_TIMEOUT_SECONDS=10
LLM_READ_TIMEOUT_SECONDS=600
```

### TTS 配置

**Qwen3-TTS**:

```bash
QWEN_TTS_BACKEND=openai
QWEN_TTS_BASE_URL=http://host.docker.internal:8091/v1
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_TTS_VOICE=aiden
QWEN_TTS_SAMPLE_RATE=24000
```

**CosyVoice**:

```bash
COSYVOICE_TTS_BACKEND=openai
COSYVOICE_TTS_BASE_URL=http://host.docker.internal:8092/v1
COSYVOICE_TTS_MODEL=cosyvoice3
COSYVOICE_TTS_VOICE=default
```

### VAD 配置

```bash
# VAD 类型 (energy | silero)
VAD_BACKEND=silero

# Silero VAD ONNX 模型路径
VAD_MODEL_PATH=/models/silero_vad.onnx

# VAD 阈值
VAD_THRESHOLD=0.5

# Energy VAD 参数
VAD_RMS_THRESHOLD=450.0
VAD_FRAME_MS=20
VAD_PADDING_MS=120
```

---

## 🔌 客户端集成

### iot-mimo-ai-nokeys 客户端

**认证配置**:

```python
# 所有 WebSocket 连接使用统一的 Bearer Token
headers = {
    "Authorization": f"Bearer {api_key}"
}

# ASR
ws_asr = await websockets.connect(
    "wss://api.example.com:18443/asr/v1/realtime",
    additional_headers=headers
)

# LLM
ws_llm = await websockets.connect(
    "wss://api.example.com:18443/ws/v1/chat/completions",
    additional_headers=headers
)

# TTS
ws_tts = await websockets.connect(
    "wss://api.example.com:18443/tts/v1/realtime",
    additional_headers=headers
)
```

**优化建议**:
- 实施 TTS 连接池(参考优化方案文档)
- 实施 LLM 缓存优化
- 实施记忆检索并发化
- 参考 `针对性优化方案-按优先级分阶段.md` 进行性能优化

---

## 📊 监控和健康检查

### 健康检查端点

每个服务都提供 `/healthz` 端点:

```bash
# ASR 服务
curl http://localhost:8011/healthz

# LLM 服务
curl http://localhost:8012/healthz

# TTS 服务
curl http://localhost:8013/healthz
```

### 日志

**Docker Compose**:

```bash
# 查看所有服务日志
docker compose logs -f

# 查看特定服务日志
docker compose logs -f asr
docker compose logs -f llm-wss
```

**Systemd**:

```bash
# 查看 ASR 服务日志
sudo journalctl -u asr-ws.service -f

# 查看最近 100 行
sudo journalctl -u asr-ws.service -n 100
```

### Prometheus 指标

添加 Prometheus 监控(可选):

```python
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
requests_total = Counter('requests_total', 'Total requests', ['service'])

# 延迟
request_duration = Histogram('request_duration_seconds', 'Request duration')

# 活跃连接
active_connections = Gauge('active_connections', 'Active WebSocket connections')
```

---

## 🐛 故障排查

### 问题 1: 认证失败

**症状**: WebSocket 连接立即关闭,错误码 4401

**原因**: API Key 不匹配

**解决**:
```bash
# 检查环境变量
echo $WSS_API_KEY

# 检查客户端 Authorization header
# 确保格式为: "Bearer <key>" 或直接 "<key>"
```

### 问题 2: 模型加载失败

**症状**: 服务启动缓慢或推理超时

**原因**: 模型路径错误或 GPU 内存不足

**解决**:
```bash
# 检查模型路径
ls -la $ASR_MODEL

# 检查 GPU 内存
nvidia-smi

# 调整并发限制
MAX_CONCURRENT_REQUESTS=1
```

### 问题 3: LLM 连接超时

**症状**: LLM 请求超时

**原因**: 超时配置过短或模型服务未启动

**解决**:
```bash
# 检查 LLM 服务
curl http://127.0.0.1:18080/v1/models

# 调整超时
LLM_READ_TIMEOUT_SECONDS=900

# 检查网络连接
ping 127.0.0.1
```

### 问题 4: Docker 网络问题

**症状**: 容器内无法访问 `host.docker.internal`

**原因**: Docker 网络配置问题

**解决**:
```yaml
# docker-compose.yml 中添加
extra_hosts:
  - "host.docker.internal:host-gateway"
```

---

## 📚 API 参考

### ASR WebSocket

**Endpoint**: `/asr/v1/realtime`

**协议**: 参考 `demo1/protocol.md`

**示例**: 参考 `demo1/test_asr_wss.py`

### LLM WebSocket

**Endpoint**: `/ws/v1/chat/completions`

**协议**: Chat Completions over WebSocket (SSE → WebSocket 转换)

**消息格式**:
```json
{
  "model": "Qwen3.6-35B-A3B",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": true
}
```

### TTS WebSocket

**Endpoint**: `/tts/v1/realtime` (Qwen) 或 `/tts/cosyvoice` (CosyVoice)

**协议**: 参考 `demo1/protocol.md`

**示例**: 参考 `demo1/test_tts_wss.py`

---

## 🔐 安全最佳实践

1. **使用高熵 API Key**:
   ```bash
   openssl rand -base64 32
   ```

2. **启用 TLS**:
   - 在生产环境中,Nginx 必须使用 HTTPS/WSS
   - 配置有效的 TLS 证书

3. **限制访问**:
   ```nginx
   # 只允许特定 IP 访问
   allow 192.168.1.0/24;
   deny all;
   ```

4. **定期轮换 API Key**:
   - 建议每 90 天轮换一次
   - 使用脚本自动化轮换流程

5. **监控异常请求**:
   - 记录所有认证失败
   - 设置告警阈值

---

## 📈 性能优化

参考 `针对性优化方案-按优先级分阶段.md` 获取完整的性能优化指南。

**快速优化清单**:

- [ ] 实施 TTS 连接池(阶段 0)
- [ ] 实施 LLM 缓存优化(阶段 0)
- [ ] 实施 Kotoba Whisper 预热(阶段 0)
- [ ] 实施记忆检索并发化(阶段 0)
- [ ] 实施预测性句子切分(阶段 1)
- [ ] 实施打断快速路径(阶段 1)
- [ ] 部署 vLLM(阶段 2)
- [ ] 实施 ASR 动态批处理(阶段 2)

---

## 🆙 升级指南

### 从 v1.0 升级到 v2.0

**步骤 1: 备份配置**:
```bash
cp .env .env.v1.backup
```

**步骤 2: 更新代码**:
```bash
git pull origin main
```

**步骤 3: 更新配置**:
```bash
# 统一 API Key
WSS_API_KEY=<your-key>  # 替换原来的 WSS_API_KEY 和 LLM_WSS_API_KEY

# 添加 LLM 配置
LLM_BACKEND=openai
LLM_BASE_URL=http://127.0.0.1:18080/v1
LLM_MODEL=Qwen3.6-35B-A3B

# 其他配置保持不变(向后兼容)
```

**步骤 4: 重启服务**:
```bash
docker compose down
docker compose up -d
```

**步骤 5: 验证**:
```bash
# 测试 ASR
python demo1/test_asr_wss.py

# 测试 LLM (如果适用)
# 测试 TTS
python demo1/test_tts_wss.py
```

---

## 📞 支持和贡献

**问题反馈**: 请在 GitHub Issues 中提交

**文档**: 
- [README.md](README.md) - 项目概览
- [demo1/protocol.md](demo1/protocol.md) - WebSocket 协议详解
- [针对性优化方案-按优先级分阶段.md](针对性优化方案-按优先级分阶段.md) - 性能优化指南

**贡献**: 欢迎提交 Pull Request

---

## 📝 更新日志

### v2.0 (2026-09-24)

**重大改进**:
- ✅ 统一认证机制(所有服务共享 `WSS_API_KEY`)
- ✅ 统一推理架构(LLM Backend 模式)
- ✅ 简化配置管理(分层配置系统)

**新增功能**:
- `auth.py` - 统一认证中间件
- `LLMBackend` - LLM 统一推理架构
- 分层配置系统(`CommonConfig`/`ASRConfig`/`TTSConfig`/`LLMConfig`)

**改进**:
- 配置环境变量减少 60%+
- 更清晰的配置结构
- 向后兼容旧配置

### v1.0 (2026-09-23)

**初始版本**:
- ASR WebSocket 服务(Kotoba Whisper/Faster Whisper)
- TTS WebSocket 服务(Qwen TTS/CosyVoice)
- LLM WebSocket Bridge
- Docker Compose 部署
- Nginx TLS 网关

---

**文档版本**: v2.0  
**最后更新**: 2026-09-24  
**作者**: Claude (Fable 5)  
**项目**: sts_ws_server
