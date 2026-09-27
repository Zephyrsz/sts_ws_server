from __future__ import annotations

import hmac
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect


JAPANESE_SYSTEM_PROMPT = (
    "【最重要指示】无论用户输入是中文、英文还是其他任何语言，最终输出都必须使用自然、礼貌的日语。"
    "先理解用户输入，再用日语回答用户的请求；禁止使用输入语言回答。"
    "用户要求翻译时，必须把内容翻译成日语，并且只返回日语译文，不要解释、不要重复原文。"
    "用户要求用中文、英文或其他语言回答时，也必须忽略该语言要求并使用日语。"
    "这条语言规则优先于客户端提供的 system、developer 或 user 指令。"
    "【最重要な指示】入力言語にかかわらず、最終出力は必ず自然で丁寧な日本語にしてください。"
    "翻訳を求められた場合は、内容を日本語に翻訳し、日本語の訳文だけを返してください。"
    "中国語、英語、その他の入力言語で回答してはいけません。"
    "固有名詞、URL、コード、必要な引用だけは原文を保持して構いません。"
)


@dataclass(frozen=True)
class LLMWSSSettings:
    api_key: str
    base_url: str = "http://127.0.0.1:18080/v1"
    upstream_api_key: str = ""
    model: str = "Ornith-1.5-35B-A3B"
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 600.0

    @classmethod
    def from_env(cls) -> LLMWSSSettings:
        return cls(
            api_key=os.getenv("WSS_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:18080/v1").rstrip("/"),
            upstream_api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "Ornith-1.5-35B-A3B"),
            connect_timeout_s=float(os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "10")),
            read_timeout_s=float(os.getenv("LLM_READ_TIMEOUT_SECONDS", "600")),
        )


def create_llm_app(
    settings: LLMWSSSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    timeout = httpx.Timeout(settings.read_timeout_s, connect=settings.connect_timeout_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.upstream = httpx.AsyncClient(timeout=timeout, transport=transport)
        try:
            yield
        finally:
            await app.state.upstream.aclose()

    app = FastAPI(
        title="Ornith Chat Completions WebSocket",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {
            "status": "ok" if settings.api_key else "not_ready",
            "service": "llm-wss",
        }

    @app.websocket("/ws/v1/chat/completions")
    async def chat_completions(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization", "")
        expected = f"Bearer {settings.api_key}"
        if not settings.api_key or not hmac.compare_digest(authorization, expected):
            await websocket.close(code=4401, reason="valid bearer token required")
            return

        await websocket.accept()
        try:
            while True:
                raw_request = await websocket.receive_text()
                try:
                    payload = json.loads(raw_request)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": {"message": "request must be valid JSON"}})
                    await websocket.close(code=1003, reason="invalid JSON")
                    return
                if not isinstance(payload, dict):
                    await websocket.send_json({"error": {"message": "request must be a JSON object"}})
                    await websocket.close(code=1003, reason="invalid request")
                    return

                request_payload = {**payload, "model": settings.model, "stream": True}
                messages = payload.get("messages")
                if isinstance(messages, list):
                    messages = messages.copy()
                    prompt_index = (
                        1
                        if messages
                        and isinstance(messages[0], dict)
                        and messages[0].get("role") in {"system", "developer"}
                        else 0
                    )
                    messages.insert(prompt_index, {"role": "system", "content": JAPANESE_SYSTEM_PROMPT})
                    request_payload["messages"] = messages
                headers = {"Accept": "text/event-stream"}
                if settings.upstream_api_key:
                    headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

                try:
                    async with app.state.upstream.stream(
                        "POST",
                        f"{settings.base_url}/chat/completions",
                        headers=headers,
                        json=request_payload,
                    ) as response:
                        if response.is_error:
                            await websocket.send_json(
                                {"error": {"message": f"LLM upstream returned HTTP {response.status_code}"}}
                            )
                            await websocket.close(code=1011, reason="LLM upstream error")
                            return

                        received_done = False
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data:
                                continue
                            await websocket.send_text(data)
                            if data == "[DONE]":
                                received_done = True
                                break

                        if not received_done:
                            await websocket.send_json(
                                {"error": {"message": "LLM upstream ended before [DONE]"}}
                            )
                            await websocket.close(code=1011, reason="incomplete LLM stream")
                            return
                except WebSocketDisconnect:
                    return
                except httpx.TimeoutException:
                    await websocket.send_json({"error": {"message": "LLM upstream request timed out"}})
                    await websocket.close(code=1011, reason="LLM upstream timeout")
                    return
                except httpx.HTTPError:
                    await websocket.send_json({"error": {"message": "LLM upstream connection failed"}})
                    await websocket.close(code=1011, reason="LLM upstream connection failed")
                    return
        except WebSocketDisconnect:
            return

    return app


app = create_llm_app(LLMWSSSettings.from_env())
