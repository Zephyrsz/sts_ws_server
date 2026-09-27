# 第二台服务器 WSS 并发连接与推理约束

本文档记录第二台服务器 `8.211.122.183` 当前公网 WSS 服务的并发行为和限制。配置对应 `pj_gpusever2/`，不适用于第一台服务器 `47.87.138.166`。

## 当前服务

| 服务 | 公网 WSS 路径 | 本机上游 | 运行进程数 |
| --- | --- | --- | ---: |
| ASR | `/asr/v1/realtime` | `127.0.0.1:8011` | 1 |
| Qwen3 TTS | `/tts/v1/realtime` | `127.0.0.1:18100` | 1 |
| Kokoro TTS | `/tts/kokoro` | `127.0.0.1:18101` | 1 |
| Sarashina TTS | `/tts/sarashina` | `127.0.0.1:18102` | 1 |
| Ornith LLM | `/ws/v1/chat/completions` | `127.0.0.1:18444` | 1 bridge + 1 llama.cpp |

公网入口统一为：

```text
wss://8.211.122.183:18443
```

## Nginx 连接限制

Nginx 在 `/etc/nginx/nginx.conf` 的 `http {}` 中定义两个共享连接计数区：

```nginx
limit_conn_zone $binary_remote_addr zone=wss_per_ip:10m;
limit_conn_zone $server_name zone=wss_global:10m;
```

五个 WebSocket location 都使用：

```nginx
limit_conn wss_per_ip 32;
limit_conn wss_global 64;
limit_conn_status 503;
limit_conn_log_level notice;
```

含义：

- 同一个公网源 IP 最多保持 32 个 WSS 连接；
- 所有公开 WSS 路径合计最多保持 64 个连接；
- 连接额度由 ASR、Qwen、Kokoro、Sarashina、LLM 共同使用；
- 超过限制时，WebSocket 升级阶段返回 HTTP `503`；
- 这些参数限制的是已建立的连接，不是每秒请求数，也不是模型推理并发。

例如，某个 IP 已经建立 32 个连接后，即使全局 64 个连接额度尚有剩余，该 IP 的第 33 个连接仍会被拒绝。

## Nginx worker 与文件描述符

当前 `/etc/nginx/nginx.conf`：

```nginx
worker_processes 20;

events {
    worker_connections 768;
}
```

Nginx systemd override：

```ini
# /etc/systemd/system/nginx.service.d/limits.conf
[Service]
LimitNOFILE=524288
```

当前生效的 Nginx worker 数量为 20，文件描述符 soft/hard limit 均为 `524288`。

理论上，20 个 worker × 768 个连接槽位约为 15,360 个 Nginx 连接槽位。反向代理 WebSocket 通常同时占用客户端到 Nginx、Nginx 到上游两条连接，因此实际可承载量还会受到上游文件描述符、内存和网络状态影响。由于全局 WSS 限制只有 64，当前实际瓶颈不是 Nginx worker 数量。

## 各服务推理并发

### Qwen3 TTS

systemd 环境设置：

```ini
Environment=MAX_CONCURRENT_REQUESTS=1
```

Qwen WSS adapter 可以接受多个客户端连接，但同一时间只允许一个 TTS 推理进入模型。其他请求会在 adapter 的 `asyncio.Semaphore` 上等待。

### Kokoro TTS

systemd 环境设置：

```ini
Environment=MAX_CONCURRENT_REQUESTS=1
```

行为与 Qwen3 TTS 相同：多个 WSS 连接可以同时存在，但单个 Kokoro 进程同时只执行一个模型推理。

### Sarashina TTS

Sarashina WSS 服务在代码中固定使用：

```python
asyncio.Semaphore(1)
```

因此多个客户端可以连接，但 Sarashina 同时只执行一个推理任务，其余请求等待信号量。

### ASR

第二台 `/etc/sts-ws-server/asr.env` 当前设置：

```dotenv
MAX_CONCURRENT_REQUESTS=1
```

多个 ASR 客户端可以建立会话并上传音频；每次 `input_audio.commit` 触发的模型转写受单个推理信号量限制，同时只执行一个转写任务。

### Ornith LLM

LLM WSS bridge 本身没有显式的应用层 semaphore，可以同时维护多个客户端连接，并为每个请求建立到 llama.cpp 的上游流式请求。

llama.cpp 当前启动参数包含：

```text
--parallel 1
```

因此底层主要按一个生成序列运行。多个 LLM 客户端可以连接，但模型生成并发能力当前按 1 个序列配置，其他请求由 llama.cpp 等待或排队。

## 连接并发与推理并发的区别

```text
连接并发：客户端可以同时保持多少条 WSS 长连接
推理并发：同一时刻有多少个请求真正占用模型进行推理
```

当前配置是：

| 层级 | 当前限制 |
| --- | ---: |
| 单公网 IP 的 WSS 连接 | 32 |
| 所有 WSS 路径合计连接 | 64 |
| Qwen3 TTS 推理 | 1 |
| Kokoro TTS 推理 | 1 |
| Sarashina TTS 推理 | 1 |
| ASR 推理 | 1 |
| Ornith llama.cpp 生成序列 | 1 |

因此“支持多个客户端连接”不代表“每个客户端都能同时占用 GPU 推理”。当前设计是允许多个连接，保护模型推理资源，并让后续请求等待。

## 超时与空闲连接

Nginx 当前配置：

```nginx
proxy_read_timeout 600s;
proxy_send_timeout 600s;
```

这两个参数是代理读写空闲超时，不是连接总寿命。客户端、Nginx 和 Uvicorn 之间仍有 WebSocket Ping/Pong 活动时，健康连接可以继续保持。

Uvicorn 当前使用默认 WebSocket 心跳：

```text
--ws-ping-interval 20s
--ws-ping-timeout 20s
```

网络断开或客户端失去响应时，通常约 20 至 40 秒可以检测并清理。健康但长期不发送业务消息的连接不会仅因为没有 `response.create` 而自动关闭；如果需要此行为，应在应用层增加业务 idle timeout。

## 当前配置的适用范围

当前配置适合以下场景：

- 多个客户端保持 WSS 连接；
- 每个模型以单并发方式保护 GPU 显存和推理稳定性；
- 超过公网连接上限时快速返回 `503`；
- 客户端自行实现重试退避和请求排队。

大量客户端同时提交任务时，后续请求可能等待模型信号量，并受到 Nginx `600s` 代理超时影响。当前应用尚未为每个模型设置独立的最大等待队列长度或排队超时。

## 运行时检查命令

查看 Nginx worker 和文件描述符限制：

```bash
systemctl show nginx -p MainPID -p LimitNOFILE -p LimitNOFILESoft
pgrep -af 'nginx: worker process'
```

查看公网 18443 的已建立连接数：

```bash
ss -Hnt state established '( sport = :18443 )' | wc -l
```

查看各模型上游监听：

```bash
ss -lntp | grep -E '18443|18100|18101|18102|8011|18444'
```

查看服务状态：

```bash
systemctl is-active nginx \
  asr-ws.service \
  sts-ws-qwen.service \
  kokoro-ws.service \
  sarashina-ws.service \
  llm-wss.service \
  llama-server.service
```

## 调整原则

如果需要提高模型推理并发，应先单独压测对应 GPU、显存和端到端延迟，再调整 `MAX_CONCURRENT_REQUESTS` 或 llama.cpp 的 `--parallel`。不建议仅通过增加 Uvicorn `--workers` 提升并发，因为每个 worker 可能复制模型实例或绕过单进程 semaphore，导致显存和 GPU 负载突然增加。

