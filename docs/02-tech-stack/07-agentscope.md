# AgentScope 入门与工程实践

> AgentScope 是面向 Agent 与 Multi-Agent 应用的开发框架。学习它的重点不是记住某个版本的类名，而是理解消息、Agent、Tool、Memory、Pipeline 和运行时如何协作。

!!! note "版本提示"
    AgentScope 的版本演进较快。本文提供架构方法和概念性代码。实际安装与 API 请以官方文档中与你锁定版本对应的示例为准。

## 1. 适用场景

- 中文 Agent 项目快速原型。
- 多 Agent 消息协作。
- 需要工具、记忆、格式化和可观测能力。
- 教学和研究中搭建清晰的角色系统。

如果只是简单问答或固定 RAG，直接使用模型 SDK 往往更轻。

## 2. 核心抽象

| 抽象 | 作用 |
|:---|:---|
| Message | 发送者、角色、内容及元数据 |
| Agent | 接收消息、调用模型或工具、产生响应 |
| Model | 对底层模型服务的封装 |
| Toolkit | 注册并执行工具 |
| Memory | 保存短期对话或长期信息 |
| Pipeline | 组织顺序、并行和条件执行 |
| Formatter | 把消息转换为模型输入格式 |

## 3. 环境与配置

建议单独创建环境并锁定版本：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install agentscope
python -m pip freeze > requirements-lock.txt
```

密钥通过环境变量或 Secret Manager 提供，不能写入仓库。

```python
import os

api_key = os.environ["MODEL_API_KEY"]
```

## 4. 单 Agent 最小流程

框架无关的逻辑如下：

```python
class Assistant:
    def __init__(self, model, system_prompt, toolkit):
        self.model = model
        self.system_prompt = system_prompt
        self.toolkit = toolkit
        self.history = []

    def reply(self, message):
        self.history.append(message)
        decision = self.model.generate(
            system=self.system_prompt,
            messages=self.history,
            tools=self.toolkit.schemas(),
        )
        return self.handle(decision)
```

在 AgentScope 中应将对应职责交给框架的 Agent、Model、Message 和 Toolkit 组件，而不是把所有逻辑塞进 Prompt。

## 5. Message 设计

消息不应只有字符串：

```json
{
  "name": "researcher",
  "role": "assistant",
  "content": "找到三条证据",
  "metadata": {
    "task_id": "T-100",
    "artifact_ids": ["A-1", "A-2"],
    "confidence": 0.86
  }
}
```

结构化元数据便于：

- 路由。
- 追踪。
- 权限判断。
- 结果合并。
- 评测。

## 6. 工具注册

工具需要明确 schema 和错误语义：

```python
def search_papers(query: str, limit: int = 5) -> dict:
    """搜索论文元数据，不下载全文。"""
    if not query.strip():
        return {"ok": False, "error": "empty_query"}
    if not 1 <= limit <= 20:
        return {"ok": False, "error": "invalid_limit"}
    return {"ok": True, "items": []}
```

工具层应负责：

- 输入验证。
- 用户权限。
- 超时与重试。
- 结果裁剪。
- Secret 隔离。
- 审计日志。

## 7. Memory

Memory 至少分为：

### 短期对话

保存当前任务必要的最近消息。

### 任务状态

保存计划、步骤、工具观察和产物引用。

### 长期记忆

保存稳定偏好或可复用知识。写入长期记忆前应检查：

- 是否真实。
- 是否稳定。
- 是否获得用户许可。
- 是否包含敏感信息。
- 何时过期。

不要把完整对话无条件写入向量库。

## 8. 多 Agent 协作示例

```text
User
 -> Coordinator
    -> Researcher: 生成证据表
    -> Analyst: 比较方案
    -> Reviewer: 按 rubric 检查
 -> Coordinator: 汇总
```

Coordinator 只传递任务需要的信息：

```json
{
  "objective": "比较方案 A/B",
  "input_artifacts": ["requirements.json"],
  "expected_output": "comparison.json",
  "max_steps": 6
}
```

## 9. Pipeline 选择

| 类型 | 适合 |
|:---|:---|
| Sequential | 有明确前后依赖 |
| Parallel | 独立研究、独立生成候选 |
| Conditional | 根据分类或验证结果路由 |
| Loop | 有明确停止条件的迭代 |

循环必须有：

- 最大轮数。
- 质量阈值。
- 重复检测。
- 成本预算。
- 失败出口。

## 10. 可观测性

每次运行记录：

```json
{
  "run_id": "R-1",
  "agent": "researcher",
  "model": "configured-model",
  "input_tokens": 1200,
  "output_tokens": 340,
  "latency_ms": 1800,
  "tool_calls": 2,
  "status": "completed"
}
```

Trace 不应记录明文密钥、密码、身份证或完整隐私数据。

## 11. 测试

### Tool 单测

- 参数边界。
- 超时。
- 空结果。
- 权限拒绝。

### Agent 评测

- 工具选择。
- 格式正确率。
- 最大轮数。
- 任务成功率。
- 安全违规。

### Multi-Agent 评测

- 委派准确率。
- 重复任务率。
- 汇总遗漏率。
- 相对单 Agent 的收益。

## 12. 常见问题

### Agent 不停对话

定义终止消息、最大轮数和验收器。

### Agent 使用错误工具

缩小工具职责，补充 Use When / Do Not Use，增加 few-shot。

### 消息越来越长

只保留必要历史，把产物存外部存储并传引用。

### 多 Agent 结果冲突

要求来源和结构化证据，由 Reviewer 按 rubric 判断。

## 面试表达

> 使用 AgentScope 时，我会把模型、消息、工具、记忆和协作协议分层。Agent 只获得最小工具权限，任务通过结构化消息传递，长产物通过引用共享。循环有最大轮数和验收条件，并用 trace、工具单测和多次运行评测稳定性。

## 延伸阅读

- [AgentScope 官方文档](https://doc.agentscope.io/)
- [AgentScope GitHub](https://github.com/agentscope-ai/agentscope)
- [AgentScope 论文](https://arxiv.org/abs/2402.14034)
