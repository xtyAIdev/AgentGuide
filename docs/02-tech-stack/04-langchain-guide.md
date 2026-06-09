# LangChain 与 LangGraph 实战指南

> LangChain 提供模型、Prompt、Tool、Retriever 等应用组件；LangGraph 提供有状态、可循环、可恢复的 Agent 编排。新项目不要从“堆 Chain”开始，而应先画清状态、节点和转移条件。

!!! note "版本说明"
    LangChain 生态更新较快。本文重点讲稳定的架构概念。安装包名称和少量 API 可能随版本变化，实际项目请锁定依赖并以官方文档为准。

## 1. 什么时候使用

适合：

- 需要快速接入多个模型或向量库。
- 需要标准化 Tool、Message、Retriever。
- 需要显式状态图、循环、人工审批和恢复。
- 团队愿意接受框架抽象并建立版本测试。

不适合：

- 只有一次模型调用。
- 三五十行原生 SDK 就能清楚完成。
- 团队无法控制依赖升级。
- 对每个底层调用都要求极致定制。

## 2. 核心组件

| 组件 | 作用 |
|:---|:---|
| Chat Model | 接收消息并生成文本或工具调用 |
| Prompt Template | 参数化模型输入 |
| Output Parser | 把输出变成结构化对象 |
| Tool | 让模型执行受控动作 |
| Retriever | 根据查询返回相关文档 |
| Runnable | 可组合执行单元 |
| StateGraph | 用状态和边描述执行流程 |
| Checkpointer | 保存图状态，实现恢复和长任务 |

## 3. 最小模型调用

不同模型提供商使用不同集成包，但调用模式通常类似：

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="你是严谨的技术助理。"),
    HumanMessage(content="用三点解释什么是 RAG。"),
]

response = model.invoke(messages)
print(response.content)
```

### 生产要求

- 模型名放配置，不硬编码。
- 设置超时和重试。
- 记录 request id、延迟、Token 和费用。
- 敏感字段在 trace 前脱敏。
- 对结构化输出做 schema 验证。

## 4. Prompt + 结构化输出

自由文本不适合作为工作流控制信号。应定义 schema：

```python
from typing import Literal
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    route: Literal["search", "calculate", "answer"]
    reason_summary: str = Field(max_length=200)
    confidence: float = Field(ge=0, le=1)
```

模型结构化输出的结果仍要验证：

- 枚举值是否合法。
- 参数是否越界。
- 业务权限是否允许。
- 低置信度是否转人工。

## 5. Tool 设计

```python
from langchain_core.tools import tool

@tool
def search_orders(customer_id: str, limit: int = 10) -> dict:
    """查询客户订单。

    仅用于只读查询。limit 必须在 1 到 50 之间。
    """
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    return {"items": [], "total": 0}
```

一个好 Tool 应包含：

- 动词明确的名称。
- 清晰的使用时机。
- 严格输入 schema。
- 小而结构化的返回值。
- 稳定错误码。
- 权限等级。

模型选择 Tool 不代表系统必须执行。真正执行前仍需策略层授权。

## 6. Retriever 与 RAG

Retriever 的接口可抽象为：

```python
docs = retriever.invoke("如何降低 RAG 幻觉？")
```

但完整 RAG 不等于 `retriever | prompt | model`。还需要：

- Query Rewrite。
- 元数据过滤。
- 混合检索。
- Rerank。
- 上下文预算。
- 引用映射。
- 无证据拒答。

详见 [RAG 全链路实战](20-rag-full-pipeline.md)。

## 7. 为什么使用 LangGraph

普通线性 Chain：

```text
输入 -> 检索 -> 生成 -> 输出
```

Agent 往往需要：

```text
输入
 -> 判断是否调用工具
 -> 调用工具
 -> 更新状态
 -> 再判断
 -> 人工审批
 -> 完成或失败
```

这本质上是带循环和条件分支的状态机。

## 8. 设计 State

State 是图中节点共享的业务数据：

```python
from typing import TypedDict

class AgentState(TypedDict):
    task: str
    messages: list
    evidence: list[dict]
    step_count: int
    status: str
    error: str | None
```

设计原则：

- 只保存后续步骤真正需要的数据。
- 大文件存对象存储，State 只留引用。
- 字段含义稳定、可序列化。
- 区分用户输入、工具结果和派生产物。
- 不把全部日志塞进消息历史。

## 9. 节点与路由

概念性示例：

```python
def decide(state: AgentState) -> dict:
    if state["step_count"] >= 6:
        return {"status": "failed", "error": "max_steps"}
    decision = decide_next_action(state)
    return {"status": decision["route"]}

def run_tool(state: AgentState) -> dict:
    result = execute_authorized_tool(state)
    return {
        "evidence": [*state["evidence"], result],
        "step_count": state["step_count"] + 1,
    }

def route(state: AgentState) -> str:
    return state["status"]
```

构图的核心思想：

```text
START -> decide
decide --tool--> run_tool -> decide
decide --answer--> answer -> END
decide --approval--> human_review
decide --failed--> END
```

## 10. Checkpoint 与恢复

长任务必须能从中断点恢复。Checkpointer 应保存：

- thread/run id。
- 当前节点。
- State 快照。
- 已执行工具的幂等键。
- 等待人工输入的原因。

恢复时不能盲目重放高风险工具。写操作应使用幂等键：

```text
idempotency_key = run_id + task_id + tool_name
```

## 11. Human-in-the-loop

需要审批的典型动作：

- 发送邮件。
- 删除文件。
- 修改生产数据库。
- 付款或下单。
- 发布公开内容。

审批页面至少展示：

- Agent 想做什么。
- 参数与目标对象。
- 为什么需要该动作。
- 预期影响。
- 可修改、批准或拒绝。

## 12. 错误处理

| 错误 | 策略 |
|:---|:---|
| 超时 | 有限次指数退避 |
| 限流 | respect retry-after |
| 参数错误 | 返回模型可修复的结构化错误 |
| 权限不足 | 不重试，转人工或终止 |
| 服务故障 | 熔断、降级 |
| 模型格式错误 | schema repair，限制次数 |

不要用统一的“重试三次”处理所有错误。

## 13. 测试策略

### 节点单测

```python
def test_route_stops_at_max_steps():
    state = {
        "task": "x",
        "messages": [],
        "evidence": [],
        "step_count": 6,
        "status": "running",
        "error": None,
    }
    update = decide(state)
    assert update["status"] == "failed"
```

### 图级测试

- 正常路径。
- 工具失败路径。
- 审批拒绝路径。
- Checkpoint 恢复。
- 最大步数终止。
- 重复写操作不产生副作用。

## 14. 项目目录建议

```text
app/
├─ graph.py
├─ state.py
├─ nodes/
│  ├─ planner.py
│  ├─ tools.py
│  └─ answer.py
├─ policies/
│  └─ permissions.py
├─ prompts/
├─ evals/
└─ tests/
```

Prompt、Tool、Policy、Graph 不应全部写在一个文件里。

## 15. 选型建议

| 需求 | 建议 |
|:---|:---|
| 一次模型调用 | 原生模型 SDK |
| 简单线性 RAG | Runnable 或普通函数 |
| 有循环和条件分支 | LangGraph |
| 长任务和恢复 | LangGraph + 持久 Checkpoint |
| 高度定制运行时 | 自研小框架或更底层 SDK |

## 面试表达

> 我把 LangChain 当作模型、工具和检索组件层，把 LangGraph 当作有状态 Agent runtime。设计时先定义 State、节点、条件边、终止条件和恢复策略，再选择框架 API。对写操作增加幂等键和人工审批，对所有节点做单测，对完整图做轨迹回归。

## 延伸阅读

- [LangChain 官方文档](https://docs.langchain.com/)
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
