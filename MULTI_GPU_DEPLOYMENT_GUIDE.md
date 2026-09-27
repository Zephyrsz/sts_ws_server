# 单服务器多GPU扩展部署方案

> **版本**: 1.0  
> **日期**: 2026-09-26  
> **适用环境**: L20-8-hk (8×NVIDIA L20)  
> **目标**: 支持多个WebSocket客户端并行连接，实现横向扩展

---

## 📋 目录

1. [架构概览](#架构概览)
2. [设计原则](#设计原则)
3. [部署架构](#部署架构)
4. [配置文件](#配置文件)
5. [部署脚本](#部署脚本)
6. [运维操作](#运维操作)
7. [监控告警](#监控告警)
8. [故障排查](#故障排查)
9. [性能优化](#性能优化)
10. [扩展路线图](#扩展路线图)

---

## 架构概览

### 当前状态

**单GPU部署**:
- GPU 0: Qwen3-TTS (vLLM) + WSS适配器
- 并发限制: `MAX_CONCURRENT_REQUESTS=1`
- 单点故障: GPU故障导致服务不可用
- 性能瓶颈: 无法支持高并发请求

**问题分析**:
- 8张GPU卡仅使用1张，资源利用率低
- 客户端排队等待，响应时间长
- 无负载均衡，无法水平扩展

### 目标架构

**多GPU负载均衡部署**:
```
外部客户端 (wss://47.87.138.166:18443)
         ↓
    Nginx TLS网关 (TCP 18443)
         ↓ [least_conn负载均衡]
    ┌────┴────┬────────┬────────┐
    ↓         ↓        ↓        ↓
GPU0-TTS  GPU1-TTS GPU2-TTS GPU3-TTS (TTS服务组)
18100     18101    18102    18103
  ↓         ↓        ↓        ↓
vLLM:8091 vLLM:8191 vLLM:8291 vLLM:8391
    
ASR服务组 (独立扩展)
    ┌────┴────┐
    ↓         ↓
GPU4-ASR  GPU5-ASR
18200     18201

预留资源
GPU6-7: LLM/其他服务
```

**核心优势**:
- **高可用**: 单个GPU故障不影响整体服务
- **高并发**: 4×TTS + 2×ASR = 最多支持 24-32 并发连接
- **负载均衡**: Nginx自动分发请求到最少连接的实例
- **水平扩展**: 可根据负载动态启停GPU实例

---

## 设计原则

### 1. GPU隔离

**原则**: 一个进程绑定一张GPU，避免显存竞争

```bash
# 每个服务通过 CUDA_VISIBLE_DEVICES 绑定专用GPU
CUDA_VISIBLE_DEVICES=0 vllm serve ...  # GPU 0
CUDA_VISIBLE_DEVICES=1 vllm serve ...  # GPU 1
```

**优势**:
- 显存隔离，不会相互影响
- 故障隔离，单GPU崩溃不影响其他
- 性能可预测，无资源抢占

### 2. 服务分层

**分层架构**:
```
Layer 1: Nginx (TLS终止 + 负载均衡 + 限流)
Layer 2: WSS适配器 (协议转换 + 请求验证)
Layer 3: 模型服务 (vLLM推理 / Faster-Whisper)
```

**职责分离**:
- **Nginx**: 处理TLS、认证、限流、路由
- **WSS适配器**: WebSocket协议、输入校验、PCM流转发
- **模型服务**: 专注于推理性能

### 3. 无状态设计

**原则**: 每个请求独立，不依赖会话状态

- 客户端可连接到任意GPU实例
- 实例可随时重启，不影响其他连接
- 支持滚动升级和灰度发布

### 4. 可观测性

**三大支柱**:
- **日志**: 结构化JSON日志，包含request_id, gpu_id
- **指标**: Prometheus metrics (延迟、吞吐、GPU利用率)
- **追踪**: 请求级别的调用链追踪

---

## 部署架构

### GPU资源分配

| GPU ID | 服务类型 | 监听端口 | vLLM端口 | 并发数 | 备注 |
|--------|---------|---------|---------|--------|------|
| 0 | Qwen TTS | 18100 | 8091 | 4 | 主TTS实例 |
| 1 | Qwen TTS | 18101 | 8191 | 4 | TTS扩展1 |
| 2 | Qwen TTS | 18102 | 8291 | 4 | TTS扩展2 |
| 3 | Qwen TTS | 18103 | 8391 | 4 | TTS扩展3 |
| 4 | ASR | 18200 | - | 6 | ASR主实例 |
| 5 | ASR | 18201 | - | 6 | ASR扩展 |
| 6 | LLM (预留) | 18300 | 18080 | 1 | Ornith模型 |
| 7 | 备用 | - | - | - | 弹性扩展 |

**总并发能力**:
- TTS: 4 GPU × 4 并发 = 16 并发请求
- ASR: 2 GPU × 6 并发 = 12 并发请求
- 峰值: 28 并发连接

### 网络拓扑

```
公网入口: 47.87.138.166:18443 (TLS)
         ↓
内网负载均衡: 172.19.0.1:18100-18103 (TTS)
              172.19.0.1:18200-18201 (ASR)
         ↓
本地模型服务: 127.0.0.1:8091-8391 (vLLM)
```

**安全隔离**:
- 公网仅暴露 18443 端口
- 内网端口仅限 Docker bridge 访问
- 模型服务仅绑定 localhost

---

## 配置文件

### 1. Systemd服务模板

#### TTS vLLM服务模板

创建 `/etc/systemd/system/qwen3-tts-vllm@.service`:

```ini
[Unit]
Description=Qwen3 TTS vLLM Service (GPU %i)
After=network.target
Documentation=https://github.com/vllm-project/vllm

[Service]
Type=simple
User=roger
Group=roger
WorkingDirectory=/app/sts_ws_server

# GPU隔离
Environment=CUDA_VISIBLE_DEVICES=%i
Environment=HF_HUB_OFFLINE=1
Environment=PYTHONUNBUFFERED=1

# Python环境
Environment=PATH=/app/sts_ws_server/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

# vLLM启动命令
ExecStart=/app/sts_ws_server/venv/bin/vllm serve \
  /app/sts_ws_server/models/tts/qwen3 \
  --deploy-config /app/sts_ws_server/venv/lib/python3.13/site-packages/vllm_omni/deploy/qwen3_tts.yaml \
  --omni \
  --host 127.0.0.1 \
  --port 8%i91 \
  --trust-remote-code \
  --enforce-eager \
  --served-model-name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90

# 故障重启
Restart=on-failure
RestartSec=15
TimeoutStartSec=300
TimeoutStopSec=60

# 资源限制
LimitNOFILE=1048576
LimitNPROC=32768

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qwen3-tts-vllm-gpu%i

[Install]
WantedBy=multi-user.target
```

**说明**:
- `%i`: systemd实例标识符，用于GPU ID
- 端口计算: GPU 0→8091, GPU 1→8191, GPU 2→8291
- `--gpu-memory-utilization 0.90`: 使用90%显存

#### TTS WSS适配器模板

创建 `/etc/systemd/system/sts-ws-qwen@.service`:

```ini
[Unit]
Description=STS WebSocket Qwen TTS Adapter (GPU %i)
After=qwen3-tts-vllm@%i.service
Requires=qwen3-tts-vllm@%i.service
BindsTo=qwen3-tts-vllm@%i.service

[Service]
Type=simple
User=roger
Group=roger
WorkingDirectory=/app/sts_ws_server/service-src

# 加载敏感配置
EnvironmentFile=/app/sts_ws_server/wss.env

# 服务配置
Environment=SERVICE_NAME=qwen-tts
Environment=QWEN_TTS_BACKEND=openai
Environment=QWEN_TTS_BASE_URL=http://127.0.0.1:8%i91/v1
Environment=QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
Environment=QWEN_TTS_VOICE=aiden
Environment=QWEN_TTS_SAMPLE_RATE=24000

# 性能配置
Environment=MAX_CONCURRENT_REQUESTS=4
Environment=MAX_TEXT_CHARS=10000
Environment=MODEL_TIMEOUT_SECONDS=300
Environment=TTS_AUDIO_CHUNK_BYTES=4096

# 启动命令
ExecStart=/app/sts_ws_server/venv/bin/uvicorn \
  sts_ws_server.app:app \
  --host 172.19.0.1 \
  --port 1810%i \
  --workers 1 \
  --log-level info

# 故障重启
Restart=on-failure
RestartSec=5
TimeoutStartSec=60

# 资源限制
LimitNOFILE=65535

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sts-ws-qwen-gpu%i

[Install]
WantedBy=multi-user.target
```

**说明**:
- `BindsTo`: 绑定到vLLM服务，vLLM停止时自动停止
- 端口计算: GPU 0→18100, GPU 1→18101
- `MAX_CONCURRENT_REQUESTS=4`: 每个实例支持4并发

#### ASR服务模板

创建 `/etc/systemd/system/sts-ws-asr@.service`:

```ini
[Unit]
Description=STS WebSocket ASR Service (GPU %i)
After=network.target

[Service]
Type=simple
User=roger
Group=roger
WorkingDirectory=/app/sts_ws_server/service-src

# 加载配置
EnvironmentFile=/app/sts_ws_server/wss.env

# GPU隔离
Environment=CUDA_VISIBLE_DEVICES=%i

# 服务配置
Environment=SERVICE_NAME=asr
Environment=ASR_BACKEND=kotoba
Environment=ASR_MODEL=/app/sts_ws_server/models/asr/kotoba-whisper-v2.1
Environment=ASR_DEVICE=cuda:0

# VAD配置
Environment=VAD_BACKEND=energy
Environment=VAD_RMS_THRESHOLD=450.0
Environment=VAD_FRAME_MS=20
Environment=VAD_PADDING_MS=120

# 性能配置
Environment=MAX_CONCURRENT_REQUESTS=6
Environment=MAX_AUDIO_SECONDS=120
Environment=MODEL_TIMEOUT_SECONDS=300

# 启动命令 (端口偏移计算: GPU4→18200, GPU5→18201)
ExecStart=/bin/bash -c '\
  PORT=$((18200 + %i - 4)); \
  /app/sts_ws_server/venv/bin/uvicorn \
    sts_ws_server.app:app \
    --host 172.19.0.1 \
    --port $PORT \
    --workers 1 \
    --log-level info'

# 故障重启
Restart=on-failure
RestartSec=5
TimeoutStartSec=180

# 资源限制
LimitNOFILE=65535

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sts-ws-asr-gpu%i

[Install]
WantedBy=multi-user.target
```

### 2. Nginx配置

创建 `/etc/nginx/conf.d/multi-gpu-wss.conf`:

```nginx
# Upstream定义 - TTS后端
upstream tts_qwen_backend {
    least_conn;  # 最少连接数算法
    
    server 172.19.0.1:18100 max_fails=3 fail_timeout=30s weight=1;
    server 172.19.0.1:18101 max_fails=3 fail_timeout=30s weight=1;
    server 172.19.0.1:18102 max_fails=3 fail_timeout=30s weight=1;
    server 172.19.0.1:18103 max_fails=3 fail_timeout=30s weight=1;
    
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

# Upstream定义 - ASR后端
upstream asr_backend {
    least_conn;
    
    server 172.19.0.1:18200 max_fails=3 fail_timeout=30s weight=1;
    server 172.19.0.1:18201 max_fails=3 fail_timeout=30s weight=1;
    
    keepalive 16;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

# 限流配置
limit_req_zone $binary_remote_addr zone=perip:10m rate=100r/m;
limit_req_zone $http_authorization zone=pertoken:10m rate=200r/m;
limit_conn_zone $binary_remote_addr zone=concurrent_perip:10m;

# TLS网关
server {
    listen 18443 ssl http2;
    server_name 47.87.138.166;
    
    # TLS配置
    ssl_certificate /app/sts_ws_server/tls/mimo-model-ip.crt;
    ssl_certificate_key /app/sts_ws_server/tls/mimo-model-ip.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # 日志
    access_log /var/log/nginx/wss-access.log combined;
    error_log /var/log/nginx/wss-error.log warn;
    
    # 健康检查端点
    location = /healthz {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }
    
    # TTS WebSocket路由
    location = /tts/v1/realtime {
        # 限流
        limit_req zone=perip burst=20 nodelay;
        limit_req zone=pertoken burst=50 nodelay;
        limit_conn concurrent_perip 10;
        
        # 代理到TTS后端
        proxy_pass http://tts_qwen_backend;
        
        # WebSocket升级
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 传递头部
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        
        # 保留认证头
        proxy_set_header Authorization $http_authorization;
        
        # WebSocket超时
        proxy_connect_timeout 10s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
        
        # 禁用缓冲
        proxy_buffering off;
        proxy_request_buffering off;
        
        # 错误处理
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 2;
        proxy_next_upstream_timeout 10s;
    }
    
    # ASR WebSocket路由
    location = /asr/v1/realtime {
        # 限流
        limit_req zone=perip burst=20 nodelay;
        limit_req zone=pertoken burst=50 nodelay;
        limit_conn concurrent_perip 10;
        
        # 代理到ASR后端
        proxy_pass http://asr_backend;
        
        # WebSocket升级
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 传递头部
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header Authorization $http_authorization;
        
        # WebSocket超时
        proxy_connect_timeout 10s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
        
        # 禁用缓冲
        proxy_buffering off;
        proxy_request_buffering off;
        
        # 错误处理
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_next_upstream_tries 2;
    }
}
```

**配置说明**:
- `least_conn`: 最少连接数负载均衡，确保请求分配均匀
- `max_fails=3 fail_timeout=30s`: 3次失败后标记不可用30秒
- `keepalive 32`: 保持32个长连接到后端
- `limit_req`: 每IP 100请求/分钟，每token 200请求/分钟
- `limit_conn`: 每IP最多10个并发连接
- `proxy_next_upstream`: 自动故障转移

### 3. 环境配置文件

`/app/sts_ws_server/wss.env` (权限 0600):

```bash
# 统一认证密钥 (64位随机hex)
WSS_API_KEY=<your-64-char-random-key>

# 日志配置
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# 监控配置
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

---

## 部署脚本

### 主部署脚本

创建 `/app/sts_ws_server/deploy-multi-gpu.sh`:

```bash
#!/bin/bash
# deploy-multi-gpu.sh - 多GPU自动化部署脚本
# 用途: 部署、管理多GPU TTS和ASR服务

set -euo pipefail

# ==================== 配置 ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/app/sts_ws_server"

# GPU分配
TTS_GPU_LIST=(0 1 2 3)      # TTS使用GPU 0-3
ASR_GPU_LIST=(4 5)          # ASR使用GPU 4-5

# 端口配置
TTS_WSS_BASE_PORT=18100     # TTS WSS适配器起始端口
TTS_VLLM_BASE_PORT=8091     # TTS vLLM起始端口 (间隔100)
ASR_BASE_PORT=18200         # ASR起始端口

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 工具函数 ====================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $*"
}

check_prerequisites() {
    log "检查前置条件..."
    
    # 检查nvidia-smi
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi 未找到，请确认NVIDIA驱动已安装"
        exit 1
    fi
    
    # 检查GPU数量
    local gpu_count
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    log "检测到 $gpu_count 张GPU"
    
    if [ "$gpu_count" -lt 6 ]; then
        log_warn "GPU数量少于6张，部分服务可能无法启动"
    fi
    
    # 检查systemd
    if ! systemctl --version &>/dev/null; then
        log_error "systemd 未找到"
        exit 1
    fi
    
    # 检查Python环境
    if [ ! -f "$BASE_DIR/venv/bin/python" ]; then
        log_error "Python虚拟环境未找到: $BASE_DIR/venv"
        exit 1
    fi
    
    # 检查模型文件
    if [ ! -d "$BASE_DIR/models/tts/qwen3" ]; then
        log_error "Qwen3-TTS模型未找到"
        exit 1
    fi
    
    log_success "前置条件检查通过"
}

wait_for_service() {
    local url=$1
    local service_name=$2
    local max_wait=${3:-60}
    local interval=2
    
    log "等待 $service_name 启动 (最多${max_wait}秒)..."
    
    for ((i=0; i<max_wait; i+=interval)); do
        if curl -fsS --max-time 2 "$url" &>/dev/null; then
            log_success "$service_name 已就绪"
            return 0
        fi
        sleep $interval
    done
    
    log_error "$service_name 启动超时"
    return 1
}

# ==================== TTS部署 ====================

deploy_tts_instance() {
    local gpu_id=$1
    local wss_port=$((TTS_WSS_BASE_PORT + gpu_id))
    local vllm_port=$((TTS_VLLM_BASE_PORT + gpu_id * 100))
    
    log "部署 TTS GPU${gpu_id} (WSS:${wss_port}, vLLM:${vllm_port})"
    
    # 启动vLLM模型服务
    log "  启动 vLLM 模型服务..."
    sudo systemctl enable "qwen3-tts-vllm@${gpu_id}.service" 2>/dev/null || true
    sudo systemctl restart "qwen3-tts-vllm@${gpu_id}.service"
    
    # 等待vLLM就绪
    if wait_for_service "http://127.0.0.1:${vllm_port}/v1/models" "vLLM GPU${gpu_id}" 120; then
        # 检查模型是否加载
        local model_count
        model_count=$(curl -fsS "http://127.0.0.1:${vllm_port}/v1/models" | grep -c "Qwen3-TTS" || echo "0")
        if [ "$model_count" -gt 0 ]; then
            log_success "  vLLM GPU${gpu_id} 模型已加载"
        else
            log_warn "  vLLM GPU${gpu_id} 模型未找到"
        fi
    else
        log_error "  vLLM GPU${gpu_id} 启动失败"
        sudo journalctl -u "qwen3-tts-vllm@${gpu_id}.service" -n 50 --no-pager
        return 1
    fi
    
    # 启动WSS适配器
    log "  启动 WSS 适配器..."
    sudo systemctl enable "sts-ws-qwen@${gpu_id}.service" 2>/dev/null || true
    sudo systemctl restart "sts-ws-qwen@${gpu_id}.service"
    
    # 等待WSS适配器就绪
    sleep 3
    if wait_for_service "http://172.19.0.1:${wss_port}/healthz" "WSS GPU${gpu_id}" 30; then
        local health_status
        health_status=$(curl -fsS "http://172.19.0.1:${wss_port}/healthz" | grep -o '"status":"[^"]*"' || echo "unknown")
        log_success "  WSS GPU${gpu_id} 健康状态: $health_status"
    else
        log_error "  WSS GPU${gpu_id} 启动失败"
        sudo journalctl -u "sts-ws-qwen@${gpu_id}.service" -n 30 --no-pager
        return 1
    fi
    
    log_success "TTS GPU${gpu_id} 部署完成"
}

deploy_all_tts() {
    log "开始部署 TTS 服务集群..."
    
    local failed=0
    for gpu_id in "${TTS_GPU_LIST[@]}"; do
        if ! deploy_tts_instance "$gpu_id"; then
            ((failed++))
            log_warn "TTS GPU${gpu_id} 部署失败，继续部署其他实例"
        fi
    done
    
    if [ $failed -eq ${#TTS_GPU_LIST[@]} ]; then
        log_error "所有TTS实例部署失败"
        return 1
    elif [ $failed -gt 0 ]; then
        log_warn "TTS部署完成，但有 $failed 个实例失败"
    else
        log_success "所有TTS实例部署成功"
    fi
}

# ==================== ASR部署 ====================

deploy_asr_instance() {
    local gpu_id=$1
    local idx=$2
    local port=$((ASR_BASE_PORT + idx))
    
    log "部署 ASR GPU${gpu_id} (端口:${port})"
    
    # 启动ASR服务
    sudo systemctl enable "sts-ws-asr@${gpu_id}.service" 2>/dev/null || true
    sudo systemctl restart "sts-ws-asr@${gpu_id}.service"
    
    # 等待服务就绪 (ASR模型加载较慢)
    sleep 5
    if wait_for_service "http://172.19.0.1:${port}/healthz" "ASR GPU${gpu_id}" 180; then
        log_success "ASR GPU${gpu_id} 部署完成"
    else
        log_error "ASR GPU${gpu_id} 启动失败"
        sudo journalctl -u "sts-ws-asr@${gpu_id}.service" -n 30 --no-pager
        return 1
    fi
}

deploy_all_asr() {
    log "开始部署 ASR 服务集群..."
    
    local failed=0
    for idx in "${!ASR_GPU_LIST[@]}"; do
        local gpu_id="${ASR_GPU_LIST[$idx]}"
        if ! deploy_asr_instance "$gpu_id" "$idx"; then
            ((failed++))
            log_warn "ASR GPU${gpu_id} 部署失败，继续部署其他实例"
        fi
    done
    
    if [ $failed -eq ${#ASR_GPU_LIST[@]} ]; then
        log_error "所有ASR实例部署失败"
        return 1
    elif [ $failed -gt 0 ]; then
        log_warn "ASR部署完成，但有 $failed 个实例失败"
    else
        log_success "所有ASR实例部署成功"
    fi
}

# ==================== Nginx管理 ====================

reload_nginx() {
    log "重载 Nginx 配置..."
    
    if sudo nginx -t; then
        sudo systemctl reload nginx
        log_success "Nginx 配置已重载"
    else
        log_error "Nginx 配置测试失败"
        return 1
    fi
}

# ==================== 状态检查 ====================

show_status() {
    echo ""
    echo "========================================="
    echo "  多GPU服务状态"
    echo "========================================="
    
    echo -e "\n${BLUE}TTS 服务:${NC}"
    printf "%-10s %-20s %-20s\n" "GPU ID" "vLLM状态" "WSS状态"
    printf "%-10s %-20s %-20s\n" "------" "--------" "-------"
    
    for gpu_id in "${TTS_GPU_LIST[@]}"; do
        local vllm_status wss_status
        vllm_status=$(systemctl is-active "qwen3-tts-vllm@${gpu_id}.service" 2>/dev/null || echo "inactive")
        wss_status=$(systemctl is-active "sts-ws-qwen@${gpu_id}.service" 2>/dev/null || echo "inactive")
        
        # 颜色标记
        if [ "$vllm_status" = "active" ]; then
            vllm_status="${GREEN}active${NC}"
        else
            vllm_status="${RED}${vllm_status}${NC}"
        fi
        
        if [ "$wss_status" = "active" ]; then
            wss_status="${GREEN}active${NC}"
        else
            wss_status="${RED}${wss_status}${NC}"
        fi
        
        printf "%-10s %-30s %-30s\n" "$gpu_id" "$(echo -e $vllm_status)" "$(echo -e $wss_status)"
    done
    
    echo -e "\n${BLUE}ASR 服务:${NC}"
    printf "%-10s %-20s\n" "GPU ID" "服务状态"
    printf "%-10s %-20s\n" "------" "--------"
    
    for idx in "${!ASR_GPU_LIST[@]}"; do
        local gpu_id="${ASR_GPU_LIST[$idx]}"
        local asr_status
        asr_status=$(systemctl is-active "sts-ws-asr@${gpu_id}.service" 2>/dev/null || echo "inactive")
        
        if [ "$asr_status" = "active" ]; then
            asr_status="${GREEN}active${NC}"
        else
            asr_status="${RED}${asr_status}${NC}"
        fi
        
        printf "%-10s %-30s\n" "$gpu_id" "$(echo -e $asr_status)"
    done
    
    echo -e "\n${BLUE}GPU 显存使用:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r idx name mem_used mem_total util; do
        local usage_pct=$((mem_used * 100 / mem_total))
        printf "GPU%s: %-20s %5s/%5s MB (%3s%%) GPU利用率: %3s%%\n" \
            "$idx" "$name" "$mem_used" "$mem_total" "$usage_pct" "$util"
    done
    
    echo -e "\n${BLUE}Nginx 状态:${NC}"
    systemctl is-active nginx >/dev/null && echo -e "${GREEN}active${NC}" || echo -e "${RED}inactive${NC}"
    
    echo ""
}

# ==================== 停止服务 ====================

stop_all_services() {
    log "停止所有服务..."
    
    # 停止TTS服务
    for gpu_id in "${TTS_GPU_LIST[@]}"; do
        log "  停止 TTS GPU${gpu_id}..."
        sudo systemctl stop "sts-ws-qwen@${gpu_id}.service" 2>/dev/null || true
        sudo systemctl stop "qwen3-tts-vllm@${gpu_id}.service" 2>/dev/null || true
    done
    
    # 停止ASR服务
    for gpu_id in "${ASR_GPU_LIST[@]}"; do
        log "  停止 ASR GPU${gpu_id}..."
        sudo systemctl stop "sts-ws-asr@${gpu_id}.service" 2>/dev/null || true
    done
    
    log_success "所有服务已停止"
}

# ==================== 健康检查 ====================

health_check() {
    log "执行健康检查..."
    
    local total=0
    local healthy=0
    
    # 检查TTS
    for gpu_id in "${TTS_GPU_LIST[@]}"; do
        local wss_port=$((TTS_WSS_BASE_PORT + gpu_id))
        ((total++))
        
        if curl -fsS --max-time 3 "http://172.19.0.1:${wss_port}/healthz" | grep -q "ok"; then
            log_success "  TTS GPU${gpu_id}: 健康"
            ((healthy++))
        else
            log_error "  TTS GPU${gpu_id}: 不健康"
        fi
    done
    
    # 检查ASR
    for idx in "${!ASR_GPU_LIST[@]}"; do
        local port=$((ASR_BASE_PORT + idx))
        ((total++))
        
        if curl -fsS --max-time 3 "http://172.19.0.1:${port}/healthz" | grep -q "ok"; then
            log_success "  ASR GPU${ASR_GPU_LIST[$idx]}: 健康"
            ((healthy++))
        else
            log_error "  ASR GPU${ASR_GPU_LIST[$idx]}: 不健康"
        fi
    done
    
    echo ""
    log "健康检查完成: $healthy/$total 实例健康"
    
    if [ $healthy -eq $total ]; then
        return 0
    else
        return 1
    fi
}

# ==================== 主函数 ====================

main_deploy() {
    log "========================================"
    log "  开始多GPU部署"
    log "========================================"
    
    check_prerequisites
    
    # 部署TTS集群
    if ! deploy_all_tts; then
        log_error "TTS集群部署失败"
        exit 1
    fi
    
    # 部署ASR集群
    if ! deploy_all_asr; then
        log_error "ASR集群部署失败"
        exit 1
    fi
    
    # 重载Nginx
    if ! reload_nginx; then
        log_warn "Nginx重载失败，请手动检查配置"
    fi
    
    # 等待服务稳定
    sleep 5
    
    # 健康检查
    if health_check; then
        log_success "========================================"
        log_success "  部署完成！所有服务健康"
        log_success "========================================"
    else
        log_warn "========================================"
        log_warn "  部署完成，但部分服务不健康"
        log_warn "========================================"
    fi
    
    # 显示状态
    show_status
}

# ==================== 命令行接口 ====================

usage() {
    cat << EOF
用法: $0 <command>

命令:
  deploy       部署所有服务
  status       显示服务状态
  health       执行健康检查
  stop         停止所有服务
  restart      重启所有服务
  reload       仅重载Nginx配置
  
示例:
  $0 deploy    # 部署所有GPU实例
  $0 status    # 查看状态
  $0 health    # 健康检查
EOF
}

case "${1:-}" in
    deploy)
        main_deploy
        ;;
    status)
        show_status
        ;;
    health)
        health_check
        ;;
    stop)
        stop_all_services
        ;;
    restart)
        stop_all_services
        sleep 5
        main_deploy
        ;;
    reload)
        reload_nginx
        ;;
    *)
        usage
        exit 1
        ;;
esac
```

保存后设置权限:

```bash
chmod +x /app/sts_ws_server/deploy-multi-gpu.sh
```

---

## 运维操作

### 初次部署

1. **准备环境**

```bash
cd /app/sts_ws_server

# 确认模型文件存在
ls -lh models/tts/qwen3/model.safetensors
ls -lh models/asr/kotoba-whisper-v2.1/
```

2. **复制配置文件**

```bash
# 复制systemd服务文件
sudo cp deploy/systemd/qwen3-tts-vllm@.service /etc/systemd/system/
sudo cp deploy/systemd/sts-ws-qwen@.service /etc/systemd/system/
sudo cp deploy/systemd/sts-ws-asr@.service /etc/systemd/system/

# 复制Nginx配置
sudo cp deploy/nginx/multi-gpu-wss.conf /etc/nginx/conf.d/

# 重载systemd
sudo systemctl daemon-reload
```

3. **执行部署**

```bash
# 部署所有服务
sudo ./deploy-multi-gpu.sh deploy

# 等待部署完成（约5-10分钟）
# 查看实时日志
sudo journalctl -f -u 'qwen3-tts-vllm@*' -u 'sts-ws-qwen@*'
```

4. **验证部署**

```bash
# 检查服务状态
./deploy-multi-gpu.sh status

# 健康检查
./deploy-multi-gpu.sh health

# 测试连接
cd demo1
python test_tts_wss.py --text "你好，这是多GPU测试"
```

### 日常运维

#### 查看服务状态

```bash
# 所有服务状态
./deploy-multi-gpu.sh status

# 单个服务状态
systemctl status qwen3-tts-vllm@0.service
systemctl status sts-ws-qwen@1.service
```

#### 查看日志

```bash
# TTS vLLM日志
sudo journalctl -u qwen3-tts-vllm@0.service -n 100 -f

# TTS WSS适配器日志
sudo journalctl -u sts-ws-qwen@0.service -n 100 -f

# ASR日志
sudo journalctl -u sts-ws-asr@4.service -n 100 -f

# Nginx日志
sudo tail -f /var/log/nginx/wss-access.log
sudo tail -f /var/log/nginx/wss-error.log
```

#### 重启服务

```bash
# 重启单个TTS实例
sudo systemctl restart sts-ws-qwen@0.service
sudo systemctl restart qwen3-tts-vllm@0.service

# 重启所有服务
./deploy-multi-gpu.sh restart

# 仅重载Nginx（无需中断连接）
./deploy-multi-gpu.sh reload
```

#### 扩展GPU实例

```bash
# 启用GPU 7作为TTS实例
sudo systemctl enable qwen3-tts-vllm@7.service
sudo systemctl start qwen3-tts-vllm@7.service

# 等待模型加载
sleep 60

# 启动WSS适配器
sudo systemctl enable sts-ws-qwen@7.service
sudo systemctl start sts-ws-qwen@7.service

# 添加到Nginx upstream
# 编辑 /etc/nginx/conf.d/multi-gpu-wss.conf
# 添加: server 172.19.0.1:18107 max_fails=3 fail_timeout=30s;

# 重载Nginx
sudo nginx -t && sudo systemctl reload nginx
```

#### 缩减GPU实例

```bash
# 停止GPU 3实例
sudo systemctl stop sts-ws-qwen@3.service
sudo systemctl stop qwen3-tts-vllm@3.service

# 禁用自动启动
sudo systemctl disable sts-ws-qwen@3.service
sudo systemctl disable qwen3-tts-vllm@3.service

# 从Nginx upstream移除
# 编辑 /etc/nginx/conf.d/multi-gpu-wss.conf
# 注释或删除对应的server行

# 重载Nginx
sudo systemctl reload nginx
```

### 滚动升级

**目标**: 零停机更新服务

```bash
#!/bin/bash
# rolling-upgrade.sh - 滚动升级脚本

TTS_GPU_LIST=(0 1 2 3)

for gpu_id in "${TTS_GPU_LIST[@]}"; do
    echo "升级 GPU${gpu_id}..."
    
    # 停止服务
    sudo systemctl stop sts-ws-qwen@${gpu_id}.service
    sudo systemctl stop qwen3-tts-vllm@${gpu_id}.service
    
    # 更新代码
    cd /app/sts_ws_server
    git pull origin main
    
    # 重新安装依赖（如有必要）
    # venv/bin/pip install -r requirements.txt
    
    # 启动服务
    sudo systemctl start qwen3-tts-vllm@${gpu_id}.service
    sleep 60  # 等待模型加载
    
    sudo systemctl start sts-ws-qwen@${gpu_id}.service
    sleep 10
    
    # 健康检查
    if curl -fsS http://172.19.0.1:$((18100 + gpu_id))/healthz | grep -q "ok"; then
        echo "GPU${gpu_id} 升级成功"
    else
        echo "GPU${gpu_id} 升级失败，回滚..."
        # 回滚逻辑
        git checkout HEAD^
        sudo systemctl restart qwen3-tts-vllm@${gpu_id}.service
        sudo systemctl restart sts-ws-qwen@${gpu_id}.service
        exit 1
    fi
    
    # 间隔30秒再升级下一个实例
    sleep 30
done

echo "所有实例升级完成"
```

---

## 监控告警

### Prometheus指标

添加 Prometheus 监控，创建 `/app/sts_ws_server/src/sts_ws_server/monitoring.py`:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import multiprocess, CollectorRegistry
import os

# 多进程模式配置
if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    from prometheus_client import REGISTRY as registry

# GPU级别指标
gpu_memory_used = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory usage in bytes',
    ['gpu_id', 'service'],
    registry=registry
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id'],
    registry=registry
)

# WebSocket连接指标
active_websocket_connections = Gauge(
    'active_websocket_connections',
    'Number of active WebSocket connections',
    ['service', 'gpu_id'],
    registry=registry
)

# 请求指标
request_duration = Histogram(
    'request_duration_seconds',
    'Request processing duration',
    ['service', 'gpu_id', 'status'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
    registry=registry
)

model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_type', 'gpu_id'],
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
    registry=registry
)

request_total = Counter(
    'request_total',
    'Total number of requests',
    ['service', 'gpu_id', 'status'],
    registry=registry
)

# 错误指标
error_total = Counter(
    'error_total',
    'Total number of errors',
    ['service', 'gpu_id', 'error_type'],
    registry=registry
)
```

在 `app.py` 中添加 `/metrics` 端点:

```python
from sts_ws_server.monitoring import registry
from prometheus_client import generate_latest

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(registry),
        media_type="text/plain"
    )
```

### Prometheus配置

创建 `/etc/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # TTS服务
  - job_name: 'tts-gpu0'
    static_configs:
      - targets: ['172.19.0.1:18100']
        labels:
          service: 'tts'
          gpu: '0'
  
  - job_name: 'tts-gpu1'
    static_configs:
      - targets: ['172.19.0.1:18101']
        labels:
          service: 'tts'
          gpu: '1'
  
  - job_name: 'tts-gpu2'
    static_configs:
      - targets: ['172.19.0.1:18102']
        labels:
          service: 'tts'
          gpu: '2'
  
  - job_name: 'tts-gpu3'
    static_configs:
      - targets: ['172.19.0.1:18103']
        labels:
          service: 'tts'
          gpu: '3'
  
  # ASR服务
  - job_name: 'asr-gpu4'
    static_configs:
      - targets: ['172.19.0.1:18200']
        labels:
          service: 'asr'
          gpu: '4'
  
  - job_name: 'asr-gpu5'
    static_configs:
      - targets: ['172.19.0.1:18201']
        labels:
          service: 'asr'
          gpu: '5'
  
  # Nginx
  - job_name: 'nginx'
    static_configs:
      - targets: ['localhost:9113']  # nginx-prometheus-exporter
```

### Grafana仪表板

关键面板:

1. **GPU资源监控**
   - GPU显存使用率
   - GPU利用率
   - 温度和功耗

2. **服务健康**
   - 活跃连接数
   - 请求成功率
   - 错误率

3. **性能指标**
   - P50/P95/P99延迟
   - 吞吐量(请求/秒)
   - 模型推理时间

4. **负载均衡**
   - 每个GPU实例的请求分布
   - 连接数均衡度

### 告警规则

创建 `/etc/prometheus/alerts.yml`:

```yaml
groups:
  - name: sts_ws_server
    interval: 30s
    rules:
      # GPU显存告警
      - alert: HighGPUMemoryUsage
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu_id }} 显存使用率超过95%"
      
      # 服务不可用告警
      - alert: ServiceDown
        expr: up{job=~"tts-.*|asr-.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务 {{ $labels.job }} 不可用"
      
      # 高错误率告警
      - alert: HighErrorRate
        expr: |
          rate(error_total[5m]) / rate(request_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.service }} GPU{{ $labels.gpu_id }} 错误率超过5%"
      
      # 高延迟告警
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, 
            rate(request_duration_seconds_bucket[5m])
          ) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.service }} P95延迟超过10秒"
      
      # 负载不均衡告警
      - alert: UnbalancedLoad
        expr: |
          max(active_websocket_connections{service="tts"}) 
          - min(active_websocket_connections{service="tts"}) > 5
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "TTS负载不均衡，最大差异超过5个连接"
```

---

## 故障排查

### 常见问题

#### 问题1: vLLM启动失败

**症状**:
```
systemctl status qwen3-tts-vllm@0.service
● qwen3-tts-vllm@0.service - Qwen3 TTS vLLM Service (GPU 0)
   Loaded: loaded
   Active: failed (Result: exit-code)
```

**排查步骤**:

1. 查看详细日志
```bash
sudo journalctl -u qwen3-tts-vllm@0.service -n 200 --no-pager
```

2. 检查常见原因

**原因A: 显存不足**
```bash
# 检查GPU显存
nvidia-smi

# 解决: 降低gpu-memory-utilization
# 编辑 /etc/systemd/system/qwen3-tts-vllm@.service
# 修改 --gpu-memory-utilization 0.90 -> 0.85
```

**原因B: 模型文件缺失**
```bash
# 检查模型文件
ls -lh /app/sts_ws_server/models/tts/qwen3/

# 应包含:
# - config.json
# - model.safetensors (3.8GB)
# - speech_tokenizer/model.safetensors (682MB)
```

**原因C: CUDA不可用**
```bash
# 测试CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查NVIDIA驱动
nvidia-smi
```

#### 问题2: WSS连接被拒绝

**症状**:
```
WebSocket connection failed: 502 Bad Gateway
```

**排查步骤**:

1. 检查后端服务
```bash
# WSS适配器是否运行
systemctl status sts-ws-qwen@0.service

# vLLM是否就绪
curl http://127.0.0.1:8091/v1/models
```

2. 检查Nginx配置
```bash
# 测试配置
sudo nginx -t

# 查看Nginx错误日志
sudo tail -f /var/log/nginx/wss-error.log
```

3. 检查网络连通性
```bash
# 测试内网连接
curl http://172.19.0.1:18100/healthz

# 测试公网TLS
curl -k https://47.87.138.166:18443/healthz
```

#### 问题3: 负载不均衡

**症状**: 某个GPU实例负载明显高于其他

**排查步骤**:

1. 查看连接分布
```bash
# 查看Nginx状态
curl http://127.0.0.1:8080/nginx_status

# 查看各实例活跃连接
for port in 18100 18101 18102 18103; do
    echo "Port $port:"
    curl -s http://172.19.0.1:$port/metrics | grep active_websocket_connections
done
```

2. 检查Nginx upstream配置
```bash
# 确认使用 least_conn
grep -A 10 "upstream tts_qwen_backend" /etc/nginx/conf.d/multi-gpu-wss.conf
```

3. 检查各实例健康状态
```bash
./deploy-multi-gpu.sh health
```

**解决方法**:
```bash
# 方法1: 重启负载高的实例
sudo systemctl restart sts-ws-qwen@0.service

# 方法2: 调整权重
# 编辑 /etc/nginx/conf.d/multi-gpu-wss.conf
# server 172.19.0.1:18100 weight=2;  # 增加权重
```

#### 问题4: 模型推理超时

**症状**:
```json
{"type": "error", "message": "TTS backend request timed out"}
```

**排查步骤**:

1. 检查GPU利用率
```bash
nvidia-smi dmon -s u -c 10  # 监控10秒
```

2. 检查vLLM日志
```bash
sudo journalctl -u qwen3-tts-vllm@0.service -n 100 | grep -i "timeout\|error"
```

3. 测试直接调用vLLM
```bash
curl -X POST http://127.0.0.1:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "测试",
    "voice": "aiden",
    "response_format": "pcm"
  }' --max-time 60
```

**解决方法**:
```bash
# 增加超时时间
# 编辑 /etc/systemd/system/sts-ws-qwen@.service
# Environment=MODEL_TIMEOUT_SECONDS=600

# 重启服务
sudo systemctl daemon-reload
sudo systemctl restart sts-ws-qwen@0.service
```

### 紧急故障处理

#### 场景: 所有TTS实例崩溃

```bash
# 1. 快速重启所有TTS实例
for gpu in 0 1 2 3; do
    sudo systemctl restart qwen3-tts-vllm@${gpu}.service &
done
wait

# 等待模型加载
sleep 120

# 2. 重启WSS适配器
for gpu in 0 1 2 3; do
    sudo systemctl restart sts-ws-qwen@${gpu}.service
done

# 3. 验证恢复
./deploy-multi-gpu.sh health
```

#### 场景: GPU硬件故障

```bash
# 假设GPU 1硬件故障

# 1. 停止故障GPU服务
sudo systemctl stop sts-ws-qwen@1.service
sudo systemctl stop qwen3-tts-vllm@1.service

# 2. 从Nginx upstream移除
# 编辑 /etc/nginx/conf.d/multi-gpu-wss.conf
# 注释: # server 172.19.0.1:18101 ...

# 3. 重载Nginx
sudo nginx -t && sudo systemctl reload nginx

# 4. 如有备用GPU，启用GPU 7替代
sudo systemctl start qwen3-tts-vllm@7.service
sleep 60
sudo systemctl start sts-ws-qwen@7.service

# 添加到Nginx upstream
# server 172.19.0.1:18107 max_fails=3 fail_timeout=30s;
sudo systemctl reload nginx
```

---

## 性能优化

### 1. 并发数调优

**测试方法**:

```bash
# 压力测试脚本
cd demo1

# 逐步增加并发
for concurrency in 2 4 8 12 16 20; do
    echo "测试并发数: $concurrency"
    for i in $(seq 1 $concurrency); do
        python test_tts_wss.py --text "并发测试${i}" &
    done
    wait
    sleep 5
done
```

**优化建议**:

- **起始值**: `MAX_CONCURRENT_REQUESTS=4` (每GPU)
- **监控指标**: GPU利用率应在70-90%
- **调整规则**:
  - GPU利用率 < 60%: 增加并发数
  - GPU利用率 > 95%: 减少并发数
  - P95延迟 > 5秒: 减少并发数

### 2. 显存优化

**vLLM参数**:

```bash
# 激进配置 (显存利用率95%)
--gpu-memory-utilization 0.95

# 保守配置 (显存利用率85%)
--gpu-memory-utilization 0.85

# 启用KV cache量化
--kv-cache-dtype fp8

# 减小max_model_len
--max-model-len 2048
```

### 3. 网络优化

**Nginx调优**:

```nginx
# 增加worker进程
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 10240;
    use epoll;
    multi_accept on;
}

# TCP优化
http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    
    # 增加keepalive
    keepalive_timeout 65;
    keepalive_requests 1000;
}
```

**系统参数**:

```bash
# /etc/sysctl.conf
net.core.somaxconn = 32768
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.ip_local_port_range = 10000 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30

# 应用配置
sudo sysctl -p
```

### 4. 模型优化

**量化**:

```bash
# AWQ 4-bit量化 (需预先量化模型)
--quantization awq

# GPTQ 4-bit量化
--quantization gptq
```

**推测解码**:

```bash
# 启用推测解码 (需draft model)
--speculative-model <draft-model-path>
```

---

## 扩展路线图

### Phase 1: 当前架构 (完成)

- ✅ 多GPU负载均衡
- ✅ Nginx反向代理
- ✅ Systemd服务管理
- ✅ 基础监控

**并发能力**: 16-28 连接

### Phase 2: 监控增强 (1-2周)

- 🔲 Prometheus + Grafana
- 🔲 告警系统
- 🔲 自动化日志收集
- 🔲 性能基准测试

**目标**: 可观测性提升

### Phase 3: 智能调度 (2-3周)

- 🔲 Redis队列
- 🔲 动态负载均衡
- 🔲 GPU亲和性调度
- 🔲 请求优先级

**并发能力**: 50+ 连接

### Phase 4: 高可用 (3-4周)

- 🔲 多服务器部署
- 🔲 跨机器负载均衡
- 🔲 自动故障转移
- 🔲 滚动升级零停机

**并发能力**: 100+ 连接

### Phase 5: 云原生 (可选)

- 🔲 Kubernetes部署
- 🔲 HPA自动扩缩容
- 🔲 服务网格 (Istio)
- 🔲 多区域部署

**并发能力**: 无限扩展

---

## 附录

### A. 端口映射表

| 服务 | GPU ID | WSS端口 | vLLM端口 | 用途 |
|------|--------|---------|----------|------|
| TTS | 0 | 18100 | 8091 | 生产主实例 |
| TTS | 1 | 18101 | 8191 | 生产扩展1 |
| TTS | 2 | 18102 | 8291 | 生产扩展2 |
| TTS | 3 | 18103 | 8391 | 生产扩展3 |
| ASR | 4 | 18200 | - | 生产主实例 |
| ASR | 5 | 18201 | - | 生产扩展 |
| LLM | 6 | 18300 | 18080 | Ornith模型 |
| 备用 | 7 | 18107 | 8791 | 弹性扩展 |

### B. 配置文件清单

```
/app/sts_ws_server/
├── deploy-multi-gpu.sh              # 主部署脚本
├── wss.env                          # 环境配置 (0600)
├── models/
│   ├── tts/qwen3/                  # Qwen3-TTS模型
│   └── asr/kotoba-whisper-v2.1/    # ASR模型
└── deploy/
    ├── systemd/
    │   ├── qwen3-tts-vllm@.service
    │   ├── sts-ws-qwen@.service
    │   └── sts-ws-asr@.service
    └── nginx/
        └── multi-gpu-wss.conf

/etc/systemd/system/
├── qwen3-tts-vllm@.service
├── sts-ws-qwen@.service
└── sts-ws-asr@.service

/etc/nginx/conf.d/
└── multi-gpu-wss.conf

/etc/prometheus/
├── prometheus.yml
└── alerts.yml
```

### C. 快速参考命令

```bash
# 部署
./deploy-multi-gpu.sh deploy

# 状态检查
./deploy-multi-gpu.sh status
./deploy-multi-gpu.sh health

# 日志查看
sudo journalctl -u qwen3-tts-vllm@0 -f
sudo journalctl -u sts-ws-qwen@0 -f
sudo tail -f /var/log/nginx/wss-access.log

# 重启服务
sudo systemctl restart qwen3-tts-vllm@0
sudo systemctl restart sts-ws-qwen@0

# GPU监控
nvidia-smi dmon -s ucm -c 100
watch -n 1 nvidia-smi

# 性能测试
cd demo1 && python test_tts_wss.py --text "测试"

# 配置重载
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl daemon-reload
```

### D. 联系与支持

- **文档版本**: 1.0
- **最后更新**: 2026-09-26
- **适用版本**: sts_ws_server v2.0+
- **反馈**: 通过项目Issue提交

---

**文档结束**
