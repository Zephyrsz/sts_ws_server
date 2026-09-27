# Qwen3-ASR 模型选择分析

## 结论

对于 `sts_ws_server` 中新增 ASR WSS 服务，建议选择官方模型
[`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)，并将推理进程与
当前 TTS vLLM 进程隔离。本项目当前已采用官方 `qwen-asr==0.0.6` 的 Transformers
backend，使用独立的 `/app/sts_ws_server/venv-qwen3-asr` 环境；它与现有的“收到
`input_audio.commit` 后返回一次最终文本”协议直接匹配。未来需要多请求批处理或独立模型
服务时，再评估官方 `qwen-asr` 的 vLLM backend。

不建议把
[`handy-computer/Qwen3-ASR-1.7B-gguf`](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf)
作为生产 WSS 的首选。它是基于官方权重的第三方 `transcribe.cpp` GGUF 转换，适合不依赖
Python/CUDA 的离线本地推理；模型卡明确标注 `streaming: false`、`timestamps: none`，并且
当前不支持显式语言提示。要把它接入本项目，需要额外运行 `transcribe.cpp`，再自行编写
进程/HTTP/WSS 适配层。

## 对比

| 项目 | 官方 `Qwen/Qwen3-ASR-1.7B` | 第三方 `handy-computer/...-gguf` |
| --- | --- | --- |
| 权重/维护 | Qwen 官方仓库，Apache-2.0；模型卡列出官方 `qwen-asr` 工具链 | 第三方转换，继承 Apache-2.0；需跟随 `transcribe.cpp` 和转换仓库版本 |
| 推理后端 | 官方 Transformers 和 vLLM；官方提供 `qwen-asr-serve` | `transcribe.cpp`，需要自己编译 C++ 程序 |
| 流式 | 官方文档说明 vLLM 后端支持 streaming | 模型卡明确为 `streaming: false` |
| 语言 | 支持自动识别；官方模型支持 30 种语言和 22 种中文方言 | 可自动识别 30 种语言，但不能传显式语言提示 |
| 时间戳 | 可配合 `Qwen3-ForcedAligner-0.6B` 返回时间戳 | 模型卡明确为 `timestamps: none` |
| 文件大小/量化 | 两个 safetensors 分片约 4.70 GB；HF 元数据报告约 2.35 GB 参数数据，实际运行还需 KV/cache 和运行时内存 | 提供 Q8_0 2.19 GB、Q4_K_M 1.32 GB 等文件；量化可降低内存，但会增加转换运行时依赖 |
| 对本项目的接入 | 可在现有 ASR backend 后面增加官方 Python 或 HTTP adapter | 必须增加 `transcribe.cpp` 子进程/服务及协议解析，维护面更大 |

上述流式、语言、时间戳和量化信息来自两个 Hugging Face 模型卡；官方模型卡还给出了
`qwen-asr-serve Qwen/Qwen3-ASR-1.7B` 和 `/v1/chat/completions` 音频消息示例。

## 与当前代码的匹配关系

当前服务的 WSS 入口在
[`src/sts_ws_server/app.py`](src/sts_ws_server/app.py)；它接收 16 kHz 单声道 PCM，内部
执行 VAD，在 `input_audio.commit` 时调用 `ASRBackend.transcribe`，最后发送
`transcription.completed`。这与官方 Transformers 的离线调用天然匹配；若需要客户端看到
增量文本，再单独设计 Qwen vLLM streaming adapter 和新的事件语义。

当前
[`src/sts_ws_server/backends.py`](src/sts_ws_server/backends.py) 的 `openai` ASR backend
固定请求 `/audio/transcriptions` 并读取 JSON 的 `text` 字段。Qwen 官方 vLLM 服务则使用
`/v1/chat/completions`，在 `messages[].content[]` 中传 `audio_url`，响应文本位于
`choices[0].message.content`。Qwen 当前 vLLM 部署页也说明支持 OpenAI transcription API，
但仍需确认目标 vLLM 版本的实现；因此不能只把 `ASR_HTTP_MODEL` 改成
`Qwen/Qwen3-ASR-1.7B` 就完成接入。

## 建议部署方案

### 方案 A：独立 Qwen vLLM 服务

1. 在独立 Python 环境启动官方服务，例如：

   ```bash
   pip install -U qwen-asr[vllm]
   qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
     --gpu-memory-utilization 0.8 \
     --host 127.0.0.1 --port 8090
   ```

2. 在 `sts_ws_server` 新增 `QwenVLLMASRBackend`：把已有 PCM 封装成 16 kHz WAV，转成
     `data:audio/wav;base64,...`（或由内部服务读取受控 URL），请求
   `POST /v1/chat/completions`，解析 `choices[0].message.content`，并用
   `qwen_asr.parse_asr_output` 或等价解析逻辑提取语言和文本。若目标 vLLM 版本已验证
   `/v1/audio/transcriptions`，也可以复用 multipart WAV 形状，但应把它作为独立 backend
   配置，不要假设现有 OpenAI backend 的所有响应字段都兼容。

3. 保留现有 WSS/VAD/鉴权协议；把 `ASR_BACKEND` 扩展为 `qwen-vllm`，增加独立的
   `ASR_BASE_URL`、`ASR_HTTP_MODEL`/模型名和上游 key 配置。

4. ASR 服务使用单独 venv/systemd unit。当前仓库的 TTS 依赖固定了 vLLM 版本；不要在同一
   venv 中为了 ASR 强行覆盖 TTS 的 vLLM 依赖。Qwen 当前部署文档建议 vLLM nightly，而
   本仓库 `requirements.txt` 固定 `vllm==0.24.0`；`deploy/asr-ws.service` 虽是单独进程，
   当前仍引用共享 `/app/sts_ws_server/venv`，部署时应改为 ASR 专用环境，或先做模型加载
   与音频请求冒烟测试确认两套版本兼容。

### 方案 B：当前部署，进程内 Transformers

使用官方 Python 包的 Transformers backend，模型在 ASR 进程启动时加载。本项目当前采用
这个方案，模型路径为 `/app/sts_ws_server/models/asr/qwen3-asr-1.7b`，并通过独立
`venv-qwen3-asr` 固定 `qwen-asr==0.0.6`、`transformers==4.57.6` 和
`accelerate==1.12.0`。这个方案最贴合
当前一次性 `transcribe()` 接口，也不需要维护外部 HTTP 服务；代价是模型和
`transformers`/PyTorch 依赖直接进入 ASR 进程，吞吐和并发需要由现有 semaphore 控制。
不要把这些依赖安装到现有 TTS 环境；共享环境的 Transformers 版本与 Qwen3-ASR 不兼容。

### 什么时候选 GGUF

只有在明确要使用 CPU、Metal 或 Vulkan，并且接受离线最终结果、无时间戳、无显式语言
强制，以及自行维护 `transcribe.cpp` 服务时，才选择 GGUF。若选 GGUF，建议把它放在独立
的本地服务后面，由 WSS adapter 通过受控 IPC/HTTP 调用，避免在 FastAPI worker 内直接
管理 C++ 子进程。

## 最小验证清单

- 用中文、日文、英文各准备一段真实业务音频，比较官方 BF16/Q8 与 GGUF Q8/Q4 的 CER/WER、首包/最终延迟和峰值内存。
- 验证静音、噪声、120 秒上限、连续两次 `input_audio.commit`、语言自动识别和错误重试。
- 先确认模型服务的上游健康检查，再接入 Nginx；不要把模型端口直接暴露到公网。
- 若未来要返回增量转写，单独定义 partial/final 事件，不能把 GGUF 的离线结果伪装成流式。

## 主要来源

- Qwen 官方模型卡与 API/流式示例：<https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Qwen 官方推理仓库：<https://github.com/QwenLM/Qwen3-ASR>
- GGUF 模型卡及 `transcribe.cpp` 用法、能力标记和量化文件：<https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf>
- `transcribe.cpp` 模型文档与复现说明：<https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr-1.7b.md>
