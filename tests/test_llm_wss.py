import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sts_ws_server.llm_wss import LLMWSSSettings, create_llm_app


def make_client(transport: httpx.AsyncBaseTransport) -> TestClient:
    app = create_llm_app(
        LLMWSSSettings(
            api_key="client-secret",
            base_url="http://127.0.0.1:18080/v1",
            upstream_api_key="model-secret",
            model="Ornith-1.5-35B-A3B",
        ),
        transport=transport,
    )
    return TestClient(app)


def test_chat_websocket_forwards_streamed_text_and_tool_call_deltas() -> None:
    captured: dict[str, object] = {}
    sse_body = (
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\\"city\\\":\\\"To"}}]},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"kyo\\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode(),
        )

    with make_client(httpx.MockTransport(handle)) as client:
        with client.websocket_connect(
            "/ws/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
        ) as websocket:
            websocket.send_json(
                {
                    "model": "ignored-client-model",
                    "stream": False,
                    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Look up weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                }
            )
            frames = [websocket.receive_text() for _ in range(4)]

    assert captured["url"] == "http://127.0.0.1:18080/v1/chat/completions"
    assert captured["authorization"] == "Bearer model-secret"
    assert captured["payload"] == {
        "model": "Ornith-1.5-35B-A3B",
        "stream": True,
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    assert frames[0].startswith('{"id":"chatcmpl-1"')
    assert '"content":"Hello"' in frames[0]
    assert '"name":"get_weather"' in frames[1]
    assert '"arguments":"{\\"city\\":\\"To' in frames[1]
    assert '"arguments":"kyo\\"}"' in frames[2]
    assert frames[3] == "[DONE]"


def test_chat_websocket_rejects_connections_without_bearer_auth() -> None:
    async def handle(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unauthorized requests must not reach the model")

    with make_client(httpx.MockTransport(handle)) as client:
        with pytest.raises(WebSocketDisconnect) as error, client.websocket_connect(
            "/ws/v1/chat/completions"
        ):
            pass

    assert error.value.code == 4401


def test_chat_websocket_reports_upstream_http_errors_without_forwarding_body() -> None:
    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="private upstream diagnostic")

    with make_client(httpx.MockTransport(handle)) as client:
        with client.websocket_connect(
            "/ws/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
        ) as websocket:
            websocket.send_json({"messages": [{"role": "user", "content": "hi"}]})
            error = websocket.receive_json()

    assert error == {"error": {"message": "LLM upstream returned HTTP 503"}}
