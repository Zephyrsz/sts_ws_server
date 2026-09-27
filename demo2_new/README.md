# Server 2 WSS tests

第三方公网接入请先阅读 [WSS_CONNECTION_CONFIG.md](WSS_CONNECTION_CONFIG.md)。
该文档定义 ASR、LLM 和 Sarashina TTS 的公网地址、Bearer 鉴权、TLS 和基础帧参数；
本 README 主要说明如何运行本目录的验证脚本。

These scripts target the second server only. They use the dedicated `STS2_*`
environment variables so credentials and endpoint settings do not overlap with
`demo1`.

The gateway is `wss://8.211.122.183:18443`. The certificate in
`server2-wss.crt` is the public self-signed certificate presented by that
gateway. The private TLS key is not included here.

## Setup

Use the project's virtual environment, which must have `websocket-client`
installed. Create a private local environment file and put the server 2
`WSS_API_KEY` value in `STS2_API_KEY`:

```sh
cp demo2/test.env.example demo2/test.env
chmod 600 demo2/test.env
```

Edit `demo2/test.env` and set `STS2_API_KEY`. Then load it:

```sh
set -a
. demo2/test.env
set +a
```

## Test ASR and LLM independently

The ASR and LLM checks are also available as standalone commands. Both use the
server-2 public IP by default, verify `server2-wss.crt`, and read the bearer key
from `STS2_API_KEY`:

```sh
.venv/bin/python demo2/test_asr_wss.py \
  --pcm /path/to/16k-mono-speech.pcm \
  --language zh

.venv/bin/python demo2/test_llm_wss.py \
  --prompt '请用一句话介绍你自己。'
```

`--pcm` accepts raw mono PCM16LE at 16 kHz; `--asr-pcm` is an alias. A WAV file
is accepted when it is mono, 16-bit PCM, and 16 kHz. The ASR test prints the
recognized text, detected language, frame timing, and audio size. The LLM test
prints streamed response text and reports the first-frame latency.

Use `--base-url` to target another public gateway, `--ca-cert` to select its
certificate, or `--insecure` only for temporary TLS diagnostics.

## Test all services

Provide a recorded speech sample for ASR when checking recognition quality.

```sh
.venv/bin/python demo2/test_public_wss.py \
  --asr-pcm /path/to/16k-mono-speech.pcm \
  --output-dir demo2/output
```

This tests ASR, Qwen TTS, Kokoro TTS, and the streaming Ornith LLM endpoint.
CosyVoice is not installed on server 2, and is excluded by default. Add
`--enable-cosyvoice` only if that service is deployed later.

## Test each Japanese TTS and save WAV

Each command opens a public WSS connection to server 2, sends Japanese text,
collects the streamed PCM16LE frames, and writes a 24 kHz mono WAV file. The
API key is never read from a command-line history when it is supplied through
`STS2_API_KEY`:

```sh
.venv/bin/python demo2/test_qwen_tts_wss.py \
  --text 'こんにちは。これはQwen3の日本語音声テストです。' \
  --output demo2/output/qwen3-ja.wav

.venv/bin/python demo2/test_kokoro_wss.py \
  --text 'こんにちは。これはKokoroの日本語音声テストです。' \
  --voice jf_alpha \
  --output demo2/output/kokoro-ja.wav

.venv/bin/python demo2/test_sarashina_wss.py \
  --text 'こんにちは。これはSarashinaの日本語音声テストです。' \
  --output demo2/output/sarashina-ja.wav
```

The scripts use these public paths:

| Script | WSS path | Default voice |
| --- | --- | --- |
| `test_qwen_tts_wss.py` | `/tts/v1/realtime` | `aiden` |
| `test_kokoro_wss.py` | `/tts/kokoro` | `jf_alpha` |
| `test_sarashina_wss.py` | `/tts/sarashina` | server default |

Use `--base-url` to point at another gateway. TLS verification trusts
`demo2/server2-wss.crt`; `--insecure` disables it only for temporary
diagnostics. CosyVoice has no deployed endpoint on server 2, so there is no
CosyVoice WAV script in this directory.

The explicit LLM command above is equivalent to the older compatibility entry
point:

```sh
.venv/bin/python demo2/test1.py --prompt '请用一句话介绍你自己。'
```

TLS verification is enabled by default. `--insecure` is available only for
temporary diagnostics; normal tests trust `server2-wss.crt`.

## Endpoints

- ASR: `/asr/v1/realtime`
- Qwen TTS: `/tts/v1/realtime`
- Kokoro TTS: `/tts/kokoro`
- Sarashina Japanese TTS: `/tts/sarashina`
- Ornith LLM: `/ws/v1/chat/completions`
- Health: `https://8.211.122.183:18443/healthz`

The standalone checks map to the public endpoints as follows:

| Script | Public path | Input/output |
| --- | --- | --- |
| `test_asr_wss.py` | `/asr/v1/realtime` | 16 kHz PCM16LE or matching WAV; prints transcript |
| `test_llm_wss.py` | `/ws/v1/chat/completions` | Chat prompt; prints streamed text |

After the server has a configured Sarashina reference WAV and matching Japanese
transcript, test it with:

```sh
.venv/bin/python demo2/test_sarashina_wss.py --output demo2/sarashina-output.wav
```
