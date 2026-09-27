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
Kokoro 日语：${MIMO_WSS_BASE_URL}/tts/kokoro
```

所有连接都使用：

```http
Authorization: Bearer <上海专属API_KEY>
```

## 证书

`MIMO_CA_CERT` 必须指向公网 18443 网关当前提供的证书。`llm-wss.crt` 是 18444 内部 LLM bridge 的证书，不适用于 18443；两者即使 Subject/SAN 都包含相同 IP，也不能互换。网关证书为自签名证书，应从服务器通过可信渠道取得：

```bash
cp /path/to/trusted/mimo-model-ip.crt ./demo1/mimo-gateway-18443.crt
export MIMO_CA_CERT="$PWD/demo1/mimo-gateway-18443.crt"
```

正式环境应设置 `MIMO_CA_CERT` 并校验证书，不应使用 `--insecure` 或 `CERT_NONE`。

## 网络要求

上海后台服务器必须能够访问模型服务器 TCP 18443。模型服务器安全组建议只允许上海服务器公网出口 IP/32。

模型容器端口 8001、8002、8003，以及网关本地端口 8080 不应直接开放公网。

## 快速测试

```bash
cd /path/to/sts_ws_server
uv venv .venv
uv pip install --python .venv/bin/python -r demo1/requirements.txt
```

测试脚本不会自动激活虚拟环境，也不会修改 `sys.path`。请使用创建该环境的解释器执行脚本：

```bash
.venv/bin/python demo1/test_public_wss.py --help
```

`websocket-client` 的安装包名是 `websocket-client`，代码导入名是 `websocket`。`websockets` 和 PyPI 上的 `websocket==0.2.1` 都不能替代它。检查包和解释器必须使用同一条 Python 命令：

```bash
.venv/bin/python -c 'import sys, websocket; print(sys.executable); print(websocket.__file__); print(websocket.create_connection)'
```

如果提示 `No module named websocket`，重新安装到该解释器，而不是使用另一个环境的 `pip`：

```bash
uv pip install --python .venv/bin/python --reinstall 'websocket-client>=1.8,<2'
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

公网统一测试（ASR、Qwen3-TTS、CosyVoice3 和 Ornith LLM）：

```bash
export MIMO_WSS_BASE_URL='wss://47.87.138.166:18443'
export MIMO_API_KEY='<上海专属API_KEY>'

# 有网关 CA 证书时启用证书校验
export MIMO_CA_CERT="$PWD/mimo-model-ip.crt"
python test_public_wss.py \
  --asr-pcm ./test-ja.pcm \
  --text 'こんにちは。ミモの音声サービスです。' \
  --output-dir ./public-wss-results
```

统一测试脚本默认将 Qwen3-TTS 的语言设为 `Japanese`（模型接受的枚举值）；ASR
仍默认使用 `ja`。如需覆盖 TTS 语言，可传入 `--language`。

当前网关证书是自签名证书，只有在无法取得 CA 证书时才使用显式的不安全模式：

```bash
python test_public_wss.py --insecure --asr-pcm ./test-ja.pcm
```

单独测试某个服务时，跳过其他服务，例如只测试 LLM：

```bash
python test_public_wss.py \
  --skip-asr --skip-qwen-tts --skip-cosyvoice
```

脚本不会打印 API key 或完整响应内容。每个服务独立输出连接耗时、首帧耗时、音频或文本帧统计，并在任一服务失败时以非零状态退出。

统一脚本默认也测试 Kokoro 日语 TTS；使用 `--skip-kokoro` 可跳过该接口。

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
