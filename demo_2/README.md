# Server 2 WSS tests

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

## Test all services

The included PCM fixture is one second of silence at 16 kHz mono PCM16LE. It
checks the ASR WebSocket session and completion protocol; use `--asr-pcm` with a
recorded speech sample when checking recognition quality.

```sh
.venv/bin/python demo2/test_public_wss.py \
  --asr-pcm demo2/asr-smoke-16k.pcm \
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

To test only streaming LLM chat:

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

After the server has a configured Sarashina reference WAV and matching Japanese
transcript, test it with:

```sh
.venv/bin/python demo2/test_sarashina_wss.py --output demo2/sarashina-output.wav
```
