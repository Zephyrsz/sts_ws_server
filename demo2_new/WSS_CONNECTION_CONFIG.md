# 公网 WSS 基础连接配置

## 网关

```text
WSS_BASE_URL = wss://8.211.122.183:18443
```

所有服务都需要在 WebSocket 握手时携带：

```http
Authorization: Bearer <STS2_API_KEY>
```

TLS 使用 `demo2/server2-wss.crt` 验证。证书的 SAN 是 `8.211.122.183`，因此请使用
上面的 IP 连接。生产环境不要关闭证书校验，也不要把 API key 放在 URL、浏览器代码或日志中。

## ASR

```text
wss://8.211.122.183:18443/asr/v1/realtime
```

客户端发送 JSON：

```json
{"type":"session.update","language":"ja"}
```

然后发送二进制音频帧：

| 参数 | 值 |
| --- | --- |
| 编码 | signed PCM 16-bit little-endian (`pcm_s16le`) |
| 采样率 | 16,000 Hz |
| 声道 | 1（mono） |
| 帧建议 | 640 bytes（20 ms） |
| base64 | 不使用，直接发送 WebSocket binary frame |

一段音频结束后发送：

```json
{"type":"input_audio.commit"}
```

服务端返回：

```json
{"type":"transcription.completed","text":"こんにちは。","language":"ja","segments":[],"is_final":true}
```

## LLM

```text
wss://8.211.122.183:18443/ws/v1/chat/completions
```

客户端发送：

```json
{
  "model":"Ornith-1.5-35B-A3B",
  "messages":[{"role":"user","content":"请用一句话介绍你自己。"}],
  "stream":true,
  "max_tokens":128,
  "temperature":0,
  "chat_template_kwargs":{"enable_thinking":false}
}
```

服务端会强制使用部署的模型，并要求无论输入是什么语言都用日语回答。返回为文本
WebSocket frame，每帧是 JSON；收到文本 `[DONE]` 表示结束。内容帧示例：

```json
{"choices":[{"delta":{"content":"こんにちは。"}}]}
```

错误帧：

```json
{"error":{"message":"..."}}
```

## Sarashina TTS

```text
wss://8.211.122.183:18443/tts/sarashina
```

客户端发送：

```json
{"type":"response.create","text":"こんにちは。今日はいい天気ですね。","language":"ja"}
```

`language` 只能使用 `ja`、`jpn` 或 `japanese`；文本必须是日语，默认单次最多 10,000
字符。服务端先返回音频格式：

```json
{"type":"response.audio.started","format":"pcm_s16le","sample_rate":24000,"channels":1}
```

随后返回多个 binary PCM 音频帧，最后返回：

```json
{"type":"response.completed"}
```

音频参数为 24,000 Hz、mono、PCM16LE，不使用 base64。

## 通用错误与连接参数

- 缺少或错误的 Bearer key：握手关闭码 `4401`。
- 业务错误：JSON `{"type":"error","message":"..."}`；LLM 使用 `{"error":{"message":"..."}}`。
- ASR 单段音频默认上限 120 秒；Sarashina 单次文本默认上限 10,000 字符。
- 建议连接超时 10 秒，响应超时 180 秒；网络异常时指数退避重连。
- 浏览器原生 `WebSocket` 不能设置 `Authorization` header，应由后端/BFF 代为连接并注入 key。

健康检查：

```text
https://8.211.122.183:18443/healthz
```
