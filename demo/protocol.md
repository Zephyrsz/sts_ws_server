# Mimo ASR / TTS / CosyVoice3 WSS 协议 v2

## 通用约定

- WSS 入口：`wss://47.87.138.166:18443`；
- Header：`Authorization: Bearer <上海专属API_KEY>`；
- JSON 文本使用 UTF-8；
- 音频二进制数据不使用 base64。

## ASR

连接：

```text
wss://47.87.138.166:18443/asr/v1/realtime
```

服务端首先发送：

```json
{"type":"session.created","audio_format":"pcm_s16le","sample_rate":16000,"channels":1}
```

客户端发送：

```json
{"type":"session.update","language":"ja"}
```

随后连续发送 16 kHz、单声道、PCM S16LE 二进制音频帧。建议每帧 20 ms，即 640 字节。

一句话结束后发送：

```json
{"type":"input_audio.commit"}
```

服务端返回最终转写：

```json
{"type":"transcription.completed","text":"こんにちは。","language":"ja","segments":[],"is_final":true}
```

## Qwen3-TTS

连接：

```text
wss://47.87.138.166:18443/tts/v1/realtime
```

客户端发送：

```json
{"type":"response.create","text":"こんにちは。","voice":"Ono_Anna","language":"ja"}
```

服务端发送音频开始事件：

```json
{"type":"response.audio.started","format":"pcm_s16le","sample_rate":24000,"channels":1}
```

随后返回多个 24 kHz、单声道、PCM S16LE 音频块，最后发送：

```json
{"type":"response.completed"}
```

## CosyVoice3

连接：

```text
wss://47.87.138.166:18443/tts/cosyvoice
```

客户端首先发送：

```json
{"type":"start"}
```

服务端返回：

```json
{"type":"audio_format","sample_rate":24000,"channels":1,"format":"pcm_s16le"}
```

客户端按完整句子发送文本：

```json
{"type":"text","text":"こんにちは。"}
```

```json
{"type":"text","text":"これは日本語音声合成テストです。"}
```

输入完成后发送：

```json
{"type":"end"}
```

服务端持续返回 24 kHz、单声道、PCM S16LE 二进制音频块，最后发送：

```json
{"type":"done"}
```

错误统一为：

```json
{"type":"error","message":"..."}
```
