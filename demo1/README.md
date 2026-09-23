# Mimo ASR / Qwen3-TTS / CosyVoice3 WSS 接入包

本包用于上海后台团队访问模型服务器上的 ASR、Qwen3-TTS 和 CosyVoice3 服务。

## 统一入口

当前正式网关入口为宿主机 TCP 18443：

```text
wss://47.87.138.166:18443
```

配置环境变量：

```bash
export MIMO_WSS_BASE_URL='wss://47.87.138.166:18443'
export MIMO_API_KEY='<上海专属API_KEY>'
export MIMO_CA_CERT="$PWD/mimo-model-ip.crt"
```

旧交付文档中的 `47.236.204.226` 与当前测试使用的 `47.87.138.166` 不一致。交付前请确认 ASR/Qwen3-TTS 是否也已经迁移到 `47.87.138.166`；如果没有，只需修改 `MIMO_WSS_BASE_URL`，不要修改协议路径。

## 服务地址

```text
ASR：       ${MIMO_WSS_BASE_URL}/asr/v1/realtime
Qwen3-TTS： ${MIMO_WSS_BASE_URL}/tts/v1/realtime
CosyVoice3：${MIMO_WSS_BASE_URL}/tts/cosyvoice
```

所有连接都使用：

```http
Authorization: Bearer <上海专属API_KEY>
```

## 证书

当前网关证书为自签名证书。请从模型服务器取得真实证书并通过安全渠道交付：

```bash
sudo cp /etc/nginx/tls/mimo-model-ip.crt ./mimo-model-ip.crt
```

正式环境应设置 `MIMO_CA_CERT` 并校验证书，不应使用 `--insecure` 或 `CERT_NONE`。

## 网络要求

上海后台服务器必须能够访问模型服务器 TCP 18443。模型服务器安全组建议只允许上海服务器公网出口 IP/32。

模型容器端口 8001、8002、8003，以及网关本地端口 8080 不应直接开放公网。

## 快速测试

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install websocket-client
```

ASR 测试：

```bash
ffmpeg -y -i ./test-ja.wav -ar 16000 -ac 1 -sample_fmt s16 -f s16le ./test-ja.pcm
python test_asr_wss.py ./test-ja.pcm
```

Qwen3-TTS 测试：

```bash
python test_tts_wss.py --text 'こんにちは。ミモの音声サービスです。'
```

CosyVoice3 测试：

```bash
python test_cosyvoice3_wss.py
```

## 当前能力边界

- ASR 支持 WSS 长连接和连续 PCM 二进制帧上传。
- Qwen3-TTS 支持 WSS 请求和 PCM 音频分块返回。
- CosyVoice3 支持句子级流式输入输出。
- CosyVoice3 当前建议按完整句子发送文本；遇到 `。`、`！`、`？` 等标点后开始合成。
- CosyVoice3 当前使用部署时配置的日文参考音色。

详细协议见 [protocol.md](protocol.md)。

## 故障排查

```bash
curl -vk https://47.87.138.166:18443/healthz
```

常见状态：

- `101`：WebSocket Upgrade 成功；
- `401`：API Key 错误或缺失；
- `404`：路径未加载或请求到了错误端口；
- `502`：网关无法连接上游服务；
- TLS 证书错误：检查 `MIMO_CA_CERT` 和证书是否为服务器真实证书。
