# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-09-24

### 重大改进

#### ✅ 统一认证机制
- 所有服务(ASR/TTS/LLM)现在共享同一个 `WSS_API_KEY`
- 新增 `src/sts_ws_server/auth.py` 统一认证中间件
- 移除分散的认证逻辑,简化维护

#### ✅ 统一推理架构
- LLM 服务改为 Backend 工厂模式(与 ASR/TTS 架构一致)
- 新增 `LLMBackend` Protocol 和 `OpenAICompatibleLLMBackend` 实现
- 统一的连接池管理(`_PooledHTTPBackend`)
- 统一的超时控制和错误处理
- 新增 `create_llm_backend` 工厂函数

#### ✅ 简化配置管理
- 引入分层配置系统:
  - `CommonConfig`: 通用配置(api_key, 超时, 并发限制等)
  - `ASRConfig`: ASR 服务专用配置
  - `TTSConfig`: TTS 服务专用配置
  - `LLMConfig`: LLM 服务专用配置
- 环境变量数量减少 **60%+**
- 配置结构更清晰,更易维护
- 完全向后兼容旧配置

### 新增文件

- `src/sts_ws_server/auth.py` - 统一认证中间件
- `DEPLOYMENT_GUIDE.md` - 完整的 v2.0 部署指南
- `CHANGELOG.md` - 版本更新日志

### 改进

- `src/sts_ws_server/config.py` - 重构为分层配置系统
- `src/sts_ws_server/backends.py` - 新增 LLM Backend 支持和 JSON 导入

### 文档更新

- 完整的 v2.0 部署指南(`DEPLOYMENT_GUIDE.md`)
- 统一架构详解
- 配置管理详解
- 升级指南(v1.0 → v2.0)
- 故障排查指南
- 完整的 API 参考
- 安全最佳实践

### 性能优化文档

- `针对性优化方案-按优先级分阶段.md`
  - 阶段 0: 快速见效优化(TTS 连接池, LLM 缓存等)
  - 阶段 1: 核心性能优化
  - 阶段 2: 架构改进
  - 阶段 3: 长期演进

### Breaking Changes

无 - 完全向后兼容 v1.0 配置

### 升级说明

参考 `DEPLOYMENT_GUIDE.md` 中的升级指南部分。

主要步骤:
1. 备份现有配置
2. 更新代码
3. 统一 API Key 配置(`WSS_API_KEY`)
4. 添加 LLM 配置项
5. 重启服务
6. 验证功能

---

## [1.0.0] - 2026-09-23

### 初始版本

- ASR WebSocket 服务(Kotoba Whisper/Faster Whisper)
- TTS WebSocket 服务(Qwen TTS/CosyVoice)
- LLM WebSocket Bridge
- Docker Compose 部署
- Nginx TLS 网关
- 基础文档和示例
