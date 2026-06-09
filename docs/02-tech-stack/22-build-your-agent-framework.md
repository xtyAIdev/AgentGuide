# 从零构建最小可用 Agent 框架

> 手撕框架的目的不是重新发明 LangGraph，而是掌握 Agent runtime 的核心：消息、工具、策略、状态、循环、权限、追踪与恢复。

本文只使用 Python 标准库和抽象模型接口。

## 1. 目标架构

```text
User Task
 -> Context Builder
 -> Model Adapter
 -> Decision Parser
 -> Policy Check
 -> Tool Executor
 -> Observation
 -> Loop Controller
 -> Final Answer
```

## 2. 项目结构

```text
mini_agent/
├─ agent.py
├─ models.py
├─ tools.py
├─ policy.py
├─ store.py
├─ tracing.py
└─ tests/
```

## 3. 数据模型

```python
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

@dataclass
class Decision:
    type: Literal["tool_call", "final"]
    reason_summary: str
    tool_call: ToolCall | None = None
    answer: str | None = None

@dataclass
class Step:
    index: int
    decision: Decision
    observation: dict[str, Any] | None = None

@dataclass
class RunState:
    run_id: str
    task: str
    steps: list[Step] = field(default_factory=list)
    status: str = "running"
    final_answer: str | None = None
```

## 4. Tool Registry

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable
    risk: str = "low"

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]
```

## 5. 参数验证

生产项目应使用 JSON Schema 或 Pydantic。最小实现也要拒绝未知字段和类型错误：

```python
def validate_arguments(schema: dict, args: dict):
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for name in required:
        if name not in args:
            raise ValueError(f"missing argument: {name}")

    unknown = set(args) - set(properties)
    if unknown:
        raise ValueError(f"unknown arguments: {sorted(unknown)}")
```

## 6. Model Adapter

框架不应绑定具体模型：

```python
from typing import Protocol

class ModelAdapter(Protocol):
    def decide(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Decision:
        ...
```

不同提供商适配器负责：

- 消息格式转换。
- Tool schema 转换。
- 结构化输出解析。
- 超时与限流。
- Token 与费用统计。

## 7. Context Builder

```python
def build_messages(state: RunState) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "完成用户任务。只能调用列出的工具。"
                "不要编造工具结果。证据充分时结束。"
            ),
        },
        {"role": "user", "content": state.task},
    ]

    for step in state.steps[-6:]:
        messages.append({
            "role": "assistant",
            "content": {
                "decision": step.decision.type,
                "reason_summary": step.decision.reason_summary,
            },
        })
        if step.observation is not None:
            messages.append({
                "role": "tool",
                "content": step.observation,
            })

    return messages
```

真实项目还应做摘要、Token 预算和不可信内容隔离。

## 8. Policy Engine

```python
class PolicyDecision:
    def __init__(self, allowed: bool, needs_approval=False, reason=""):
        self.allowed = allowed
        self.needs_approval = needs_approval
        self.reason = reason

def authorize(tool: Tool, args: dict, user: dict) -> PolicyDecision:
    if tool.risk == "high":
        return PolicyDecision(
            allowed=False,
            needs_approval=True,
            reason="high-risk tool requires approval",
        )
    return PolicyDecision(allowed=True)
```

Prompt 中写“不要删除文件”不等于安全控制。权限必须在模型外执行。

## 9. Tool Executor

```python
import time

def execute_tool(tool: Tool, args: dict) -> dict:
    started = time.perf_counter()
    try:
        validate_arguments(tool.input_schema, args)
        data = tool.handler(**args)
        return {
            "ok": True,
            "data": data,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "retryable": False,
        }
```

不要直接把 Python traceback 全部喂给模型，可能泄漏路径和敏感数据。

## 10. Agent Loop

```python
class Agent:
    def __init__(self, model, tools, max_steps=8):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(self, state: RunState, user: dict) -> RunState:
        for index in range(self.max_steps):
            messages = build_messages(state)
            decision = self.model.decide(messages, self.tools.schemas())
            step = Step(index=index + 1, decision=decision)
            state.steps.append(step)

            if decision.type == "final":
                state.status = "completed"
                state.final_answer = decision.answer
                return state

            call = decision.tool_call
            tool = self.tools.get(call.name)
            policy = authorize(tool, call.arguments, user)

            if policy.needs_approval:
                state.status = "awaiting_approval"
                step.observation = {"ok": False, "reason": policy.reason}
                return state

            if not policy.allowed:
                step.observation = {"ok": False, "reason": policy.reason}
                continue

            step.observation = execute_tool(tool, call.arguments)

        state.status = "max_steps_exceeded"
        return state
```

## 11. 防循环

仅有最大步数还不够。可以检测重复动作：

```python
import json

def fingerprint(call: ToolCall) -> str:
    args = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
    return f"{call.name}:{args}"

def has_repeated_call(state: RunState, threshold=3) -> bool:
    calls = [
        fingerprint(s.decision.tool_call)
        for s in state.steps
        if s.decision.tool_call
    ]
    return len(calls) >= threshold and len(set(calls[-threshold:])) == 1
```

## 12. Checkpoint

```python
import json
from dataclasses import asdict
from pathlib import Path

def save_state(state: RunState, directory="runs"):
    path = Path(directory) / f"{state.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

真实系统使用数据库，并处理：

- 并发版本。
- 加密。
- 过期清理。
- 用户隔离。
- 写操作幂等。

## 13. Trace

每一步记录：

- run id、step id。
- 模型和 Prompt 版本。
- Decision 摘要。
- Tool 名与脱敏参数。
- 结果状态。
- Token、延迟、费用。
- Policy 决策。

Trace 是调试和评测数据，不是无限保留的聊天记录。

## 14. 测试

```python
class FakeModel:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def decide(self, messages, tools):
        return next(self.decisions)

def test_agent_stops_after_final():
    model = FakeModel([
        Decision(type="final", answer="done", reason_summary="enough")
    ])
    agent = Agent(model, ToolRegistry())
    state = RunState(run_id="r1", task="test")
    result = agent.run(state, user={"id": "u1"})
    assert result.status == "completed"
```

重点覆盖：

- 未知工具。
- 参数错误。
- 高风险审批。
- 超时。
- 重复调用。
- 最大步数。
- 恢复后不重复副作用。

## 15. 下一步扩展

按顺序增加，而不是一次做全：

1. 结构化模型输出。
2. Tool schema 验证。
3. Policy 与审批。
4. Checkpoint。
5. Trace 与 Eval。
6. Planner。
7. 并行任务。
8. Multi-Agent。

## 面试表达

> 一个最小 Agent runtime 包含模型适配器、Context Builder、Tool Registry、Policy Engine、Loop Controller、Checkpoint 和 Trace。我会保证模型只负责提出动作，参数验证、授权和执行都在确定性代码中完成，并通过最大步数、重复检测、幂等键和回归评测控制风险。

## 延伸阅读

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
