# Ornith-1.5-35B-A3B-GGUF 本地部署建议

## 1. 结论

`bartowski/Ornith-1.5-35B-A3B-GGUF` 本地部署后，可以支持以下能力：

- **Streaming Output（流式输出）**：支持。通过 `llama-server` 提供的 OpenAI 兼容接口，将 `stream` 设置为 `true` 即可。
- **Function Calling / Tool Calling（函数调用或工具调用）**：模型本身支持，但能否稳定返回标准 OpenAI `tool_calls`，取决于推理框架、聊天模板和工具调用解析器。
- **Function Calling 与 Streaming 同时开启**：可以，但客户端需要正确拼接流式返回的函数名称和参数片段。

如果使用 GGUF 版本，建议采用较新的 `llama.cpp`。如果更重视 Function Calling 的稳定性、标准兼容性和生产部署能力，建议使用原始 FP8/BF16 模型配合 vLLM。

---

## 2. Streaming Output

### 2.1 支持情况

GGUF 模型通过 `llama-server` 启动后，可以使用 OpenAI 兼容的 `/v1/chat/completions` 接口进行流式输出。

### 2.2 启动示例

直接通过 Hugging Face 仓库加载推荐的 `Q4_K_M` 量化版本：

```bash
llama-server \
  -hf bartowski/Ornith-1.5-35B-A3B-GGUF:Q4_K_M \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 999
```

如果已经手工下载 GGUF 文件：

```bash
llama-server \
  -m ./Ornith-1.5-35B-A3B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 999
```

如果当前 `llama.cpp` 版本和模型文件支持 MTP 推测解码，可以尝试加入：

```bash
--spec-type draft-mtp
```

完整示例：

```bash
llama-server \
  -hf bartowski/Ornith-1.5-35B-A3B-GGUF:Q4_K_M \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  --spec-type draft-mtp
```

> 注意：不同版本的 `llama.cpp` 参数可能发生变化。正式部署前应使用 `llama-server --help` 检查当前版本是否支持对应参数。

### 2.3 流式调用示例

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.5-35B-A3B",
    "messages": [
      {
        "role": "user",
        "content": "请简单介绍一下风冷和液冷的区别"
      }
    ],
    "stream": true,
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 1024
  }'
```

返回形式通常为 SSE 数据流：

```text
data: {"choices":[{"delta":{"content":"风"}}]}
data: {"choices":[{"delta":{"content":"冷"}}]}
data: {"choices":[{"delta":{"content":"和"}}]}
...
data: [DONE]
```

### 2.4 Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local"
)

stream = client.chat.completions.create(
    model="Ornith-1.5-35B-A3B",
    messages=[
        {
            "role": "user",
            "content": "介绍一下本地部署大模型的优缺点"
        }
    ],
    stream=True,
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 3. Function Calling

### 3.1 模型能力层面

Ornith-1.5 系列模型支持工具调用。模型能够生成类似下面的工具调用内容：

```xml
<tool_call>
{"name":"get_weather","arguments":{"city":"Portland"}}
</tool_call>
```

推理服务器需要通过工具调用解析器，将该内容转换为 OpenAI 风格的 `tool_calls`。

官方非 GGUF 部署方案通常使用：

- vLLM：`--enable-auto-tool-choice --tool-call-parser qwen3_xml`
- SGLang：`--tool-call-parser qwen3_coder`

### 3.2 llama.cpp 支持情况

较新的 `llama.cpp` Server 支持：

- OpenAI 兼容的 `/v1/chat/completions`
- Function Calling / Tool Use
- Chat Template
- SSE Streaming
- 结构化 JSON 输出

因此，可以向本地 llama-server 发送 OpenAI 风格的 `tools` 参数。

### 3.3 工具调用请求示例

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Ornith-1.5-35B-A3B",
    "messages": [
      {
        "role": "user",
        "content": "查询 Portland 当前的数据中心室外温度"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "查询指定城市当前天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string",
                "description": "城市名称"
              }
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0.2,
    "stream": false
  }'
```

理想返回结果如下：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_001",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Portland\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

---

## 4. GGUF 部署中的关键注意事项

Function Calling 不是简单地开启一个模型参数，而是由三部分共同完成：

```text
模型生成工具调用意图
        ↓
Chat Template 将 tools 定义写入提示词
        ↓
推理服务器解析模型输出并转换成标准 tool_calls
```

### 4.1 正常返回标准 `tool_calls`

这是最理想的情况，可以直接接入 LangChain、LangGraph 或 OpenAI SDK。

### 4.2 输出 `<tool_call>`，但 API 没有解析

如果模型返回类似以下内容：

```xml
<tool_call>
{"name":"get_weather","arguments":{"city":"Portland"}}
</tool_call>
```

这通常说明模型已经识别出工具调用意图，但 llama-server 没有将其转换成标准 OpenAI `tool_calls`。可以依次检查：

1. 升级 `llama.cpp` 到较新版本。
2. 检查服务器启动日志中的 Chat Template。
3. 确认 GGUF 文件是否包含正确的聊天模板。
4. 尝试显式指定兼容的 Chat Template。
5. 在应用层解析 `<tool_call>` XML 标签和内部 JSON。
6. 如果仍不稳定，改用 vLLM 的 `qwen3_xml` 解析器。

### 4.3 将工具调用当作普通文字输出

常见原因包括：

- `llama.cpp` 版本过旧。
- GGUF 内置 Chat Template 与模型训练格式不匹配。
- `tools` 没有被正确写入 Prompt。
- 量化级别过低，导致格式遵循能力下降。
- Tool Schema 太复杂或描述不清晰。
- Reasoning、Streaming 和多工具调用组合下，解析器兼容性不足。

---

## 5. Function Calling 与 Streaming 同时使用

两项能力可以同时开启：

```json
{
  "stream": true,
  "tools": [],
  "tool_choice": "auto"
}
```

但流式工具调用可能把函数名称和 `arguments` 分成多个 delta 返回，例如：

```text
delta 1: function.name = "get_weather"
delta 2: arguments = "{\"city\""
delta 3: arguments = ":\"Portland\"}"
```

客户端需要先将所有参数片段拼接完整，再执行 JSON 解析和函数调用。

### 5.1 推荐验证顺序

1. 普通非流式文本生成。
2. 普通流式输出。
3. 非流式 Function Calling。
4. Function Calling 与 Streaming 同时开启。
5. 多工具选择。
6. 多轮工具调用循环。
7. 异常参数、工具不存在和工具执行失败等容错测试。

### 5.2 生产环境建议

为了提高稳定性，可以采用下面的策略：

- 普通聊天回答使用 Streaming。
- 工具选择阶段优先使用非 Streaming。
- 工具执行完成后的最终回答再次使用 Streaming。

这种方式更容易处理 JSON 参数不完整、工具调用解析失败和重试逻辑。

---

## 6. 接入 LangChain

可以将本地 llama-server 当作 OpenAI 兼容端点：

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    city: str = Field(description="需要查询天气的城市名称")


llm = ChatOpenAI(
    model="Ornith-1.5-35B-A3B",
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
    temperature=0.2,
    streaming=True
)

llm_with_tools = llm.bind_tools([GetWeather])

response = llm_with_tools.invoke("查询 Portland 当前天气")

print(response.tool_calls)
print(response.content)
```

如果 `response.tool_calls` 为空，但 `response.content` 中出现 `<tool_call>`，通常不是 LangChain 本身的问题，而是 llama-server 没有将模型原始输出转换成标准 OpenAI `tool_calls`。

应用层可以考虑增加以下兼容逻辑：

1. 优先读取标准 `tool_calls`。
2. 如果 `tool_calls` 为空，检查 `content` 中是否存在 `<tool_call>`。
3. 提取标签内部 JSON，并进行 Schema 校验。
4. 对工具名称和参数设置白名单。
5. 工具执行失败时，将错误结果作为 `tool` 消息返回给模型，由模型决定是否修正参数并重试。

---

## 7. 部署方案建议

### 7.1 轻量本地部署

推荐组合：

```text
GGUF Q4_K_M + 较新版本 llama.cpp
```

适合场景：

- 本地代码助手。
- 单用户 Agent。
- 内部自动化工具。
- LangChain 或 LangGraph 功能验证。
- 普通流式对话服务。

### 7.2 更重视 Function Calling 稳定性

推荐组合：

```text
官方 FP8/BF16 + vLLM + qwen3_xml parser
```

参考启动命令：

```bash
vllm serve ornith-ai/Ornith-1.5-35B-A3B \
  --served-model-name Ornith-1.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

该方案可以将模型生成的 `<tool_call>` 转换成标准 OpenAI `tool_calls`，并将推理内容分离到 `reasoning_content`。实际显存需求取决于权重格式、上下文长度、KV Cache、并发量和运行时设置。

---

## 8. 量化版本选择

对于复杂的 Function Calling，不建议使用 Q2 或较低位的 Q3 量化。工具名称、JSON 参数和 XML 标签对格式准确性要求较高，低精度量化更容易出现：

- JSON 引号、括号或转义错误。
- 函数名称拼写错误。
- 必填参数遗漏。
- XML 标签不完整。
- 工具调用与普通文本混合输出。

建议：

- **最低推荐**：`Q4_K_M`
- **质量与容量更平衡**：`Q5_K_M`
- **更接近完整质量**：`Q6_K`

最终应使用自己的实际 Tool Schema 和业务请求进行评测，而不只比较普通聊天效果。

---

## 9. 最终判断

### Streaming Output

```text
GGUF + llama.cpp：成熟、直接、可靠。
```

### Function Calling

```text
模型本身支持。
GGUF + 较新版本 llama.cpp 基本可用，但需要验证 Chat Template 和 Tool Parser。
正式 Agent 服务优先考虑官方权重 + vLLM。
```

### 推荐落地路径

```text
开发验证：Q4_K_M GGUF + llama.cpp
        ↓
验证单工具和非流式 Function Calling
        ↓
验证 Function Calling + Streaming
        ↓
验证 LangChain / LangGraph 多轮工具循环
        ↓
生产部署前评估 vLLM 官方权重方案
```

---

## 10. 参考资料

- GGUF 模型页面：<https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF>
- 官方模型页面：<https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8>
- llama.cpp Server 文档：<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
