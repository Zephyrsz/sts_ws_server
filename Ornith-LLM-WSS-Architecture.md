# Ornith LLM WSS 部署架构设计

> 状态：已部署并完成端到端验证  
> 更新日期：2026-09-23  
> 范围：Thor GPU 服务器上的 Ornith-1.5-35B-A3B Chat Completions WebSocket 服务

## 1. 目标与范围

本服务在现有 TLS 网关后提供 OpenAI Chat Completions 风格的 WebSocket 接口。客户端通过一个 WebSocket 连接提交 JSON 请求；服务将请求转成上游流式 HTTP 请求，并把上游 Server-Sent Events（SSE）中的 `data:` 内容逐帧转发给客户端。

本设计覆盖 Ornith GGUF 推理进程、WebSocket bridge、外部 Nginx 网关之间的职责、接口、部署关系和运维边界。它不是 OpenAI Realtime 语音协议，也不执行工具调用。模型生成的 `tool_calls` 由客户端接收并按业务逻辑执行。

## 2. 总体架构

```mermaid
flowchart LR
    Client[业务客户端]
    Gateway[既有 Nginx TLS 网关\n公网 TCP 18443\n外部 Bearer 鉴权]
    Bridge[FastAPI / Uvicorn WSS bridge\n主机 TCP 18444\n内部 Bearer 鉴权]
    Llama[llama-server\n127.0.0.1:18080\nOpenAI 兼容 HTTP API]
    GPU[GPU 0 + GPU 2\nOrnith Q8_0 GGUF]

    Client -->|WSS /ws/v1/chat/completions| Gateway
    Gateway -->|TLS WebSocket 反向代理\n替换为内部 Bearer key| Bridge
    Bridge -->|HTTP POST /v1/chat/completions\nstream=true| Llama
    Llama --> GPU
    GPU -->|增量 token / tool_calls| Llama
    Llama -->|SSE data: ... / [DONE]| Bridge
    Bridge -->|WebSocket 文本帧| Gateway
    Gateway --> Client
```

外部入口复用服务器已有的 `voice-api-gateway`。Nginx 完成公网 TLS 和业务方 Bearer key 校验，再通过 Docker 网络可达的主机网关地址转发到 WSS bridge。llama-server 只监听 loopback，不直接暴露给 Docker 容器或公网。

## 3. 组件与部署边界

| 组件 | 部署位置 | 监听地址 | 职责 |
| --- | --- | --- | --- |
| 业务客户端 | 业务服务或应用 | 出站连接 | 建立 WSS、发送 Chat Completions JSON、拼接流式 delta、按需执行工具 |
| `voice-api-gateway` | Docker 容器，`voice-internal` 网络 | 公网 TCP `18443` | TLS 终止、校验现有业务 Bearer key、限制并代理 WebSocket |
| `llm-wss.service` | Thor 主机 systemd 用户服务 | `0.0.0.0:18444`，TLS | 校验内部 Bearer key、转发请求、SSE 到 WebSocket 帧的协议适配 |
| `llama-server.service` | Thor 主机 systemd 用户服务 | `127.0.0.1:18080`，HTTP | 加载 GGUF、运行推理、提供 OpenAI 兼容 Chat Completions API |
| Ornith GGUF | `/app/sts_ws_server/models/ornith/` | 文件 | `Ornith-1.5-35B-Q8_0.gguf`，约 37.8 GB |

模型服务和 WSS bridge 由独立的用户级 systemd unit 管理。`llm-wss.service` 依赖 `llama-server.service`；用户 linger 已开启，使用户管理器可以在无交互登录时运行服务。

## 4. 网络与端口

| 端口 | 绑定 | 协议 | 用途与访问范围 |
| --- | --- | --- | --- |
| `18443` | 公网网卡 | HTTPS/WSS | 对外正式入口，由现有 Nginx 网关处理 |
| `18444` | `0.0.0.0` | HTTPS/WSS | 主机上的 bridge；网关通过 Docker 主机网关地址访问。外部直连测试超时，应由安全组/主机防火墙阻止公网访问 |
| `18080` | `127.0.0.1` | HTTP | llama-server 内部 API，仅主机本地可访问 |

网关到 bridge 的当前上游为 `https://172.19.0.1:18444/ws/v1/chat/completions`，其中 `172.19.0.1` 是 `voice-internal` Docker 网络的主机网关地址。公网调用方应使用 `18443`，不应依赖直接访问 `18444` 或 `18080`。

## 5. 请求与响应协议

### 5.1 客户端请求

客户端建立连接：

```text
wss://47.87.138.166:18443/ws/v1/chat/completions
Authorization: Bearer <已获发的业务网关 token>
```

连接成功后发送标准 Chat Completions JSON 对象，例如：

```json
{
  "model": "Ornith-1.5-35B-A3B",
  "messages": [
    {"role": "user", "content": "Call get_weather for Tokyo."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Look up weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": true
}
```

服务端固定使用配置的模型别名 `Ornith-1.5-35B-A3B`，并强制 `stream=true`。请求中的 `tools`、`tool_choice`、messages 和其他 Chat Completions 参数会传给 llama-server。客户端传入的 `model` 和 `stream` 值不会改变服务端的模型选择和流式行为。

### 5.2 流式响应

llama-server 输出 SSE。bridge 忽略 SSE 注释和非 `data:` 行，将每个非空 `data:` 值作为一条 WebSocket 文本消息发送，包括最终的 `[DONE]`：

```text
SSE data: {"choices":[{"delta":{"content":"Hello"}}]}
    -> WebSocket text: {"choices":[{"delta":{"content":"Hello"}}]}

SSE data: [DONE]
    -> WebSocket text: [DONE]
```

工具调用采用 Chat Completions 标准的增量 `tool_calls` 格式。工具名称和参数可能拆分在多个 delta 中；客户端需要按 `tool_calls[].index` 拼接 `function.name` 和 `function.arguments`。bridge 不缓冲、不重写这些片段，也不调用外部工具。

一个 WebSocket 连接可以依次发送多条请求；bridge 在上一条上游流结束并转发 `[DONE]` 后才读取下一条请求，不提供同一连接内的并发请求复用或响应 ID 路由。

## 6. 鉴权与 TLS

1. 调用方使用当前业务方已获发的网关 Bearer token 连接公网 `18443`。
2. Nginx 按现有网关配置识别调用方；未通过校验时在 WebSocket Upgrade 前返回 HTTP 401。
3. 对通过校验的连接，Nginx 将客户端 Authorization 替换成部署专用的内部 `WSS_API_KEY`，再转发给 bridge。客户端不应获取或使用这个内部 key。
4. bridge 使用恒定时间比较校验内部 Bearer token。独立连接 bridge 且鉴权失败时，它会在接受 WebSocket 前以关闭码 `4401` 拒绝连接；客户端库可能将其显示为握手 HTTP 403。
5. 公网 TLS 在现有网关终止。网关到 bridge 当前也使用 TLS；bridge 证书是自签名证书，Nginx 配置了 SNI，但当前没有启用上游证书校验。llama-server 使用 loopback HTTP，链路不离开主机。

凭据值只保存在服务器环境文件中，本文档不记录密钥。公网调用方需要使用已有业务网关凭据，而不是内部 `WSS_API_KEY`。

## 7. 推理配置

当前推理进程使用以下配置：

| 项目 | 当前值 |
| --- | --- |
| 推理框架 | `llama.cpp` `llama-server`，CUDA build |
| 模型文件 | `/app/sts_ws_server/models/ornith/Ornith-1.5-35B-Q8_0.gguf` |
| 模型别名 | `Ornith-1.5-35B-A3B` |
| GPU | `CUDA_VISIBLE_DEVICES=0,2`，layer split，tensor split `1,1` |
| 上下文 | 32,768 tokens |
| 并发推理槽 | 1 |
| 工具模板 | 启用 `--jinja` |
| Web UI | 禁用 `--no-webui` |

该部署加载文本 GGUF，不加载仓库中的 mmproj 文件，因此不提供图像输入能力。Function calling 由模型模板和 llama-server 的 OpenAI 兼容解析实现；部署验收时已通过强制 `get_weather` 工具调用验证，观察到增量 `tool_calls`、`finish_reason=tool_calls` 和 `[DONE]`。

## 8. 超时、错误与健康检查

| 情况 | 行为 |
| --- | --- |
| 无效 JSON 或请求不是对象 | 发出 JSON `error` 帧，然后关闭连接 |
| 上游返回 HTTP 错误 | 发出只含 HTTP 状态码的通用错误，不向客户端透传上游响应正文 |
| 上游连接失败或超时 | 发出通用 `error` 帧并关闭连接 |
| 上游流没有 `[DONE]` | 发出不完整流错误并关闭连接 |
| 客户端断开 | bridge 结束当前连接处理 |

健康检查分为两层：

- `http://127.0.0.1:18080/health`：llama-server 进程就绪状态。
- `https://<host>:18444/healthz`：bridge 进程和内部 key 配置状态；它不探测上游模型是否可推理。

网关 `/healthz` 是现有网关的存活检查，不代替以上两项检查。运行状态和日志可用 `systemctl --user status`、`journalctl --user -u llama-server.service`、`journalctl --user -u llm-wss.service` 检查。服务不记录请求正文或模型输出。

## 9. 容量与运行限制

- llama-server 配置一个推理槽；超过模型服务处理能力的请求会排队或受上游资源影响。
- 网关对同一调用方的活动连接使用 `limit_conn`，当前限制为 20。此值限制连接数，不等同于 20 路并行模型推理。
- WebSocket 单帧上限为 4 MiB；bridge 上游连接超时为 10 秒、读取超时默认为 600 秒；网关代理读写超时为 3600 秒。
- 模型分配至 GPU 0 和 2。服务器上的其他 GPU 推理任务会影响共享显存、功耗和整体延迟。
- 当前 bridge 绑定 `0.0.0.0:18444`。虽然外部直连测试超时，仍应保持安全组和主机防火墙对该端口的公网入站访问关闭；公网只开放网关 `18443`。
- 网关到 bridge 的 TLS 上游当前未校验证书。若网络边界发生变化，应启用 `proxy_ssl_verify` 并配置可信 CA，或将代理和 bridge 放到具有明确访问控制的私有网络中。

## 10. 关键部署文件

| 文件 | 用途 |
| --- | --- |
| `src/sts_ws_server/llm_wss.py` | FastAPI WebSocket bridge、鉴权和 SSE 转发 |
| `deploy/ornith/llama-server.service` | Ornith llama-server 用户级 systemd unit |
| `deploy/ornith/llm-wss.service` | TLS WSS bridge 用户级 systemd unit |
| `deploy/ornith/ornith.env.example` | bridge 环境变量模板，不包含实际凭据 |
| `tests/test_llm_wss.py` | 流式文本、工具调用、鉴权和上游错误测试 |

公网网关配置位于服务器现有 voice-stack gateway 部署目录；本服务通过其中的 `/ws/v1/chat/completions` 路由接入，不负责创建第二个公网 TLS 网关。

## 11. 验收结果

- llama-server `/health` 和模型列表接口正常。
- bridge `/healthz` 正常，systemd 服务处于 active/enabled，用户 linger 已开启。
- 从网关入口完成已鉴权的实际 WebSocket function-call 流测试，得到标准增量 `tool_calls`、`finish_reason=tool_calls` 和 `[DONE]`。
- 外部无凭据访问网关 WebSocket 路径返回 401；错误 Bearer token 未能建立 WebSocket。
- LLM WSS 专项自动化测试通过。完整本地测试集的收集受测试虚拟环境缺失可选依赖 `numpy` 影响。
