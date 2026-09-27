"""统一认证中间件 - 所有 WebSocket 服务共享"""
from __future__ import annotations

from fastapi import WebSocket, status
from starlette.websockets import WebSocketState

from .config import Settings


async def authenticate_websocket(
    websocket: WebSocket,
    settings: Settings,
    service_name: str = "service"
) -> bool:
    """
    统一的 WebSocket 认证

    Args:
        websocket: WebSocket 连接
        settings: 配置对象
        service_name: 服务名称 (用于日志)

    Returns:
        bool: 认证是否成功
    """
    # 如果未配置 API Key,跳过认证
    if not settings.common.api_key:
        return True

    # 从 headers 中获取 Authorization
    auth_header = websocket.headers.get("authorization", "")

    # 支持两种格式: "Bearer <key>" 或直接 "<key>"
    if auth_header.startswith("Bearer "):
        provided_key = auth_header[7:]
    else:
        provided_key = auth_header

    # 验证 API Key
    if provided_key != settings.common.api_key:
        await _close_with_error(
            websocket,
            f"{service_name} authentication failed",
            status.WS_1008_POLICY_VIOLATION
        )
        return False

    return True


async def _close_with_error(
    websocket: WebSocket,
    message: str,
    code: int = status.WS_1008_POLICY_VIOLATION
) -> None:
    """发送错误消息并关闭连接"""
    try:
        if websocket.client_state == WebSocketState.CONNECTING:
            await websocket.accept()

        await websocket.send_json({"type": "error", "message": message})
        await websocket.close(code=code)
    except Exception:
        # 如果发送失败,直接关闭
        try:
            await websocket.close(code=code)
        except Exception:
            pass
