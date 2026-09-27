# 多 GPU 主机上的 WSS 与模型推理扩展分析

> 生成日期：2026-09-26  
> 范围：`pj_gpuserver1`（47.87.138.166）与 `pj_gpuserver2`（8.211.122.183）当前运行的 WSS 适配器、模型推理进程及多 GPU 扩展方式。  
> 本文是现网分析和扩展建议，不是可直接执行的部署脚本。

## 结论

WSS 连接并发和模型推理并发是两个独立的容量指标。增加 Nginx、Uvicorn 或 WSS 进程数量，只能增加连接接入能力；只有增加模型实例、提高模型自身的并发槽，或把请求调度到多个后端，才会增加同时推理能力。

推荐保持每台机器一个公网 WSS/Nginx 入口，在入口之后按模型类型扩展：

- Qwen3-ASR、Kotoba ASR、Kokoro、Sarashina：当前模型加载在 WSS 进程内，近期扩展单位是“一个 GPU + 一个模型进程 + 一个 WSS 进程”。
- Qwen3-TTS：WSS 通过 HTTP 调用 vLLM，适合部署多个 vLLM 实例和对应的 WSS worker，再由入口或专用调度层选择后端。
- Ornith LLM：WSS bridge 通过 HTTP 调用 `llama-server`，适合部署多个独立 `llama-server` 实例，并对 HTTP 推理请求做模型池路由。

不要把所有路径都简单地放进同一个 `least_conn` upstream。WebSocket 连接通常是长连接，Nginx 选择的是连接，而不是连接内下一条推理请求；一个客户端长时间占用的连接可能持续固定在同一个 GPU worker 上。

## 当前代码的容量边界

### 统一语音 WSS 应用

`src/sts_ws_server/app.py` 根据 `SERVICE_NAME` 创建 ASR、Qwen TTS、CosyVoice 或 Kokoro backend，并在每个进程中创建一个 `app.state.inference_limiter`。ASR 在 `input_audio.commit` 时进入该 semaphore；TTS 在整个音频流生成期间持有该 semaphore。

因此：

- `MAX_CONCURRENT_REQUESTS=1` 是单进程模型调用上限，不是整台机器上限。
- 多开进程会多出多份 semaphore，但也可能多加载模型；不能只增加 Uvicorn worker 来“免费”获得并发。
- 同一个 WSS 连接可以保持很久，但等待 semaphore 的请求仍会占用连接和上游超时预算。

相关实现：[`src/sts_ws_server/app.py`](src/sts_ws_server/app.py) 的 `create_app`、`asr_realtime`、各 TTS websocket 路由。

### ASR backend

`Qwen3ASRBackend` 和 `KotobaWhisperBackend` 都在 WSS 进程启动时加载本地模型，并各自使用 `asyncio.Lock` 包住实际转写。Qwen3-ASR 还将 `max_inference_batch_size` 固定为 1。即使进程的 semaphore 设置大于 1，backend 自身的锁仍会把推理串行化。

相关实现：[`src/sts_ws_server/backends.py`](src/sts_ws_server/backends.py) 的 `Qwen3ASRBackend`、`KotobaWhisperBackend`。

### Sarashina

Sarashina WSS 服务固定创建 `asyncio.Semaphore(1)`，因此每个 Sarashina 进程只有一个推理槽。它不是通过 `MAX_CONCURRENT_REQUESTS` 动态扩展的。

相关实现：[`src/sts_ws_server/sarashina_wss.py`](src/sts_ws_server/sarashina_wss.py)。

### LLM bridge

Ornith WSS bridge 没有应用层推理 semaphore；它为每个请求向 `llama-server` 发起流式 HTTP 请求。当前 `llama-server` 使用 `--parallel 1`，因此单个 Ornith 实例只有一个生成槽，多个请求会由模型服务排队或等待资源。

相关实现：[`src/sts_ws_server/llm_wss.py`](src/sts_ws_server/llm_wss.py)；模型服务的并发槽由 `llama-server` 参数控制。

## 两台主机的现网快照

### `pj_gpuserver2`：8× NVIDIA L20

当前服务状态均为 active，公网入口为 `18443`，内部 WSS/模型端口如下：

| GPU | 当前负载 | WSS / 模型端口 | 推理进程关系 | 观测显存 |
| ---: | --- | --- | --- | ---: |
| 0 | Qwen3-TTS vLLM；Kokoro WSS 进程也在该卡上有 CUDA 占用 | vLLM `8091`，Qwen WSS `18100`，Kokoro WSS `18101` | Qwen TTS 是 HTTP backend；Kokoro 模型在 WSS 进程内 | 约 16.8 GiB |
| 1 | Ornith `llama-server` | HTTP `18080`；LLM bridge `18444` | 两卡 layer split 的一个模型实例 | 约 18.9 GiB |
| 2 | Ornith `llama-server` | HTTP `18080`；LLM bridge `18444` | 与 GPU 1 共同承载同一个实例 | 约 17.7 GiB |
| 3 | Sarashina WSS | WSS `18102` | 模型在 WSS 进程内，固定 semaphore 1 | 约 7.4 GiB |
| 4 | Qwen3-ASR WSS | WSS `8011` | 模型在 WSS 进程内，`cuda:4`，独立 `venv-qwen3-asr` | 约 4.8 GiB |
| 5 | 当前基本空闲 | 无 | 可作为 ASR/TTS/LLM 副本候选 | 约 3 MiB |
| 6 | 当前基本空闲 | 无 | 可作为副本候选 | 约 3 MiB |
| 7 | 当前基本空闲 | 无 | 可作为副本候选 | 约 3 MiB |

第二台新增的 ASR unit 为 `/etc/systemd/system/asr-ws.service`：

- `ASR_BACKEND=qwen3`
- `ASR_MODEL=/app/sts_ws_server/models/asr/qwen3-asr-1.7b`
- `ASR_DEVICE=cuda:4`
- `ASR_COMPUTE_TYPE=bfloat16`
- `MAX_CONCURRENT_REQUESTS=1`
- 独立 Python 环境：`/app/sts_ws_server/venv-qwen3-asr`
- 单 worker Uvicorn，监听 `127.0.0.1:8011`

其 `/healthz` 返回 `{"status":"ok","service":"asr","protocols":["asr"]}`。这是一个“WSS 进程内加载 Qwen3-ASR”的部署，不是独立 HTTP ASR 模型服务。

第二台 Nginx 当前限制所有公开 WSS 路径合计 64 条连接、单公网 IP 32 条连接；这只限制连接数，不代表 64 路模型并行。

### `pj_gpuserver1`：8× NVIDIA L20

第一台当前也运行 Ornith、Qwen3-TTS、Kokoro/Sarashina 等服务，但 GPU 资源共享更明显：

- Ornith 使用 GPU 0、2，并分别占用约 18.9 GiB、17.7 GiB。
- GPU 0 同时还有 Qwen3-TTS vLLM 及其他 Python 进程；GPU 2 还有 CosyVoice 进程。
- 当前 ASR unit 使用项目的统一 `sts_ws_server.app`，监听 `172.19.0.1:8011`，历史配置为 Kotoba Whisper；它没有第二台那样的独立 `venv-qwen3-asr` Qwen3-ASR 部署。
- 第一台的资源快照显示 GPU 0、2 以外还存在其他 GPU 工作负载，因此不能把整机剩余显存直接视为 WSS 可用容量。

第一台的 Ornith 模型参数与第二台相同，都是 Q8_0、32K 上下文、`--parallel 1`。两台的主要差异是 GPU 分配、同卡竞争、systemd 托管方式和 llama.cpp 可执行文件路径，而不是模型推理参数。

## 不同服务的扩展单位

### 1. Qwen3-ASR：复制 WSS 与模型进程

当前 Qwen3-ASR 的模型直接由 ASR WSS 进程加载。要把 GPU 4 的单槽扩展到 GPU 5/6/7，需要为每个副本同时改变：

1. `CUDA_VISIBLE_DEVICES` 或 `ASR_DEVICE`，确保模型落到目标卡；
2. 独立监听端口，例如 `8012`、`8013`；
3. 独立 systemd 实例和独立日志/健康检查；
4. 每个实例的 `MAX_CONCURRENT_REQUESTS`，起始仍建议为 1；
5. Nginx 或内部 WSS 路由，使新的客户端连接能到达所有实例。

不能只复制 WSS 端口而共享同一个进程内模型；也不能通过增加 Uvicorn worker 代替 GPU 副本，因为 worker 可能复制模型并造成显存不可控。

短期适合采用 Nginx upstream 分配**不同客户端连接**。如果同一 WebSocket 连接需要提交多条并发音频任务，应在应用层做请求级调度，或要求客户端为并发任务建立多个连接。

### 2. Kokoro / Sarashina：复制完整 WSS 实例

这两个模型当前也在 WSS 进程内：

- Kokoro 受 `KOKORO_TTS_DEVICE`、模型目录和进程内锁影响；
- Sarashina 进程固定 semaphore 1，并使用独立 `venv-sarashina`。

扩展方式是每张 GPU 一个完整进程，设置不同端口和 GPU 可见性，再由入口按连接分发。每个副本内部仍从并发 1 开始压测，不要把 Nginx 连接数直接换算成模型吞吐。

### 3. Qwen3-TTS：复制模型 backend，保留 WSS 协议层

Qwen WSS adapter 使用 `QWEN_TTS_BACKEND=openai`，通过 `QWEN_TTS_BASE_URL` 调用 vLLM 的 `/v1/audio/speech`。这使它与本地模型解耦，适合：

```text
一个公网 WSS 入口
        ↓
Qwen WSS worker / 请求调度器
        ├── vLLM GPU0 : 8091
        ├── vLLM GPU5 : 8095
        └── vLLM GPU6 : 8096
```

如果每个 WSS worker 固定一个 `QWEN_TTS_BASE_URL`，则 Nginx 只能在 WebSocket 建立时选择 worker。若要求同一连接内的每个 `response.create` 动态选择后端，需要在 WSS adapter 中实现请求级模型池，并为每个后端记录 in-flight 数量、队列上限和超时。

### 4. Ornith LLM：复制 llama-server，按 HTTP 请求路由

当前两台均是一个 Ornith Q8_0 实例，分布在两张 L20 上，`--parallel 1`。两张卡的作用是容纳一个较大的模型实例和它的上下文/KV cache，不等于两路并行。

如果目标是提高并行生成能力，有三种选择：

| 方式 | 增益 | 代价 |
| --- | --- | --- |
| 提高同一实例 `--parallel` | 共享一份权重，增加生成槽 | KV cache 按槽增长，显存、延迟和吞吐需重新压测 |
| 增加一个两卡/多卡 Ornith 副本 | 独立生成槽，隔离性好 | 每个副本都要复制约 37.8 GB Q8_0 权重，需额外 GPU 组合 |
| 在一张空闲 L20 上运行单卡副本 | 直接增加一个独立后端，适合 32K/低槽位验证 | 每卡约需 35 GiB，长上下文或多槽余量较小 |

多实例只有在 LLM bridge 或 HTTP 反向代理按请求选择后端时才会提高实际并行。仅增加多个 WSS bridge、而它们都指向同一个 `18080`，不会提高 Ornith 推理并行。

## 推荐的第二台扩展顺序

当前第二台 GPU 5、6、7 空闲，建议按业务瓶颈选择，而不是一次性占满：

### 方案 A：优先扩展语音识别

适合 ASR 请求排队明显、Qwen3-ASR 质量已经满足业务的情况：

- GPU 5：Qwen3-ASR WSS 副本，`MAX_CONCURRENT_REQUESTS=1`；
- GPU 6：Qwen3-ASR WSS 副本或 Kotoba/备用 ASR；
- GPU 7：保留故障转移或压测。

这样可将 Qwen3-ASR 从一个进程内推理槽扩展为多个独立槽。每个副本的模型加载开销约为当前 GPU 4 观测到的数 GiB，但需以启动日志和峰值显存为准。

### 方案 B：优先扩展 Ornith

适合 LLM 生成排队明显、需要多个同时对话：

- 保留 GPU 1、2 的两卡 Ornith 主实例；
- 在 GPU 5 或 6 上启动一个单卡 Q8_0 Ornith 副本，先保持 `--parallel 1` 做能力和延迟验证；
- 将 LLM bridge 的上游从单一 `LLM_BASE_URL` 扩展为后端池，按请求选择健康且有容量的 `llama-server`；
- 另一张空闲卡保留给 ASR 副本或作为故障转移。

单卡副本实测能加载当前 Q8_0、32K、单槽配置，但长上下文、多槽并发仍需单独验收。

### 方案 C：优先扩展 TTS

适合 Qwen TTS 请求占用时间最长：

- 在 GPU 5/6 部署独立 vLLM Qwen3-TTS；
- 每个 vLLM 使用独立端口；
- 每个 WSS worker 设置对应的 `QWEN_TTS_BASE_URL`；
- 先把每实例 `MAX_CONCURRENT_REQUESTS` 保持为 1，再根据 GPU 利用率、首包延迟和排队时间调高。

## 路由层设计建议

### 连接级路由

适用于当前 WSS 协议和进程内模型副本：

```text
客户端 WSS 连接
       ↓
Nginx least_conn / 健康检查
       ├── ASR WSS GPU4
       ├── ASR WSS GPU5
       └── ASR WSS GPU6
```

优点是实现简单、连接状态天然固定在一个后端；缺点是长连接会造成连接数均衡而非推理负载均衡。

### 请求级路由

适用于多个 HTTP 模型后端，尤其是 Ornith 和 Qwen TTS：

```text
WSS bridge
       ↓ 每次推理请求选择后端
后端池：健康状态 + in-flight + max_concurrent
       ├── llama-server A
       ├── llama-server B
       └── vLLM C
```

请求级路由必须满足：

- 后端选择在一次流式响应开始前完成，并在整个流期间保持不变；
- 使用原子 in-flight 计数，避免多个请求同时抢占同一槽位；
- 每个后端有最大等待队列、排队超时和失败摘除时间；
- 客户端断开时释放后端占用；
- 健康检查同时覆盖 WSS adapter 和模型 HTTP `/health`/`/v1/models`。

## 并发容量的正确计算

应分别记录以下指标：

| 指标 | 含义 |
| --- | --- |
| WSS active connections | 当前保持的 WebSocket 长连接数 |
| inference in-flight | 正在实际占用模型的请求数 |
| queue wait time | 请求等待模型槽的时间 |
| time to first audio/token | 首个音频块或 token 的延迟 |
| end-to-end latency | 从请求到完成事件的总耗时 |
| GPU memory peak | 模型加载和推理峰值显存 |
| GPU utilization | 推理期间 GPU 利用率 |

例如，第二台当前公开连接上限为全局 64，但 Qwen3-ASR、Qwen TTS、Kokoro、Sarashina 各自仍可能只有一个推理槽，Ornith 也只有一个 `--parallel 1` 生成槽。此时 64 条连接主要表示“可以等待”，不表示 64 条请求同时生成。

## 部署和验证原则

1. 每个 GPU 模型实例使用明确的 `CUDA_VISIBLE_DEVICES`，并通过 `nvidia-smi` 验证实际落卡。
2. WSS adapter 和模型 HTTP 服务分别做 `/healthz`、`/health` 或 `/v1/models` 检查。
3. 先单实例压测，再增加副本；不要先提高 `MAX_CONCURRENT_REQUESTS` 或 `--parallel`。
4. 对 ASR 使用中文、日文和英文音频分别测试；对 TTS/LLM 记录首包延迟和完成延迟。
5. 检查长文本、长音频、客户端断开、后端重启和排队超时。
6. 为每个副本建立独立日志和 GPU 标签，避免只看 Nginx 连接数判断模型负载。
7. 新副本稳定后再扩大 Nginx 全局连接上限；先扩大连接上限会把压力转化为无界等待。

## 当前状态摘要

| 服务 | `pj_gpuserver1` | `pj_gpuserver2` | 当前扩展单位 |
| --- | --- | --- | --- |
| ASR | Kotoba/统一 ASR WSS，单进程 | Qwen3-ASR，独立 venv，GPU4，8011，单并发 | WSS + 本地模型 + GPU |
| Qwen TTS | WSS → vLLM | WSS → vLLM GPU0 | vLLM + WSS worker |
| Kokoro | 本地模型 WSS | 本地模型 WSS，GPU0 有占用 | WSS + 本地模型 + GPU |
| Sarashina | 独立 WSS，固定 semaphore 1 | 独立 WSS，GPU3，固定 semaphore 1 | WSS + 本地模型 + GPU |
| Ornith | 单个两卡 llama-server，`--parallel 1` | 单个两卡 llama-server，`--parallel 1` | llama-server 实例 + HTTP 请求池 |

### 最终建议

第二台的下一步优先级应由真实排队指标决定：

1. 若 ASR 排队最多，复制 Qwen3-ASR WSS/模型到 GPU5 或 GPU6；
2. 若 LLM 排队最多，在空闲 GPU 上增加独立 Ornith 实例，并改为请求级路由；
3. 若 TTS 排队最多，复制 vLLM + Qwen WSS worker；
4. 保留至少一张卡作为故障转移或新模型验证卡。

核心原则是：**WSS 负责协议和连接，模型服务负责推理容量；只有在两者之间建立正确的连接级或请求级调度，增加 GPU 副本才会转化为真实并发能力。**

