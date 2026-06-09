# Chain-of-Thought 与规划

> CoT 解决“如何形成中间推理”，规划解决“如何把目标转成可执行步骤”。在 Agent 中，最重要的不是让模型输出很长的思考，而是生成可验证、可执行、可更新的计划。

## 1. CoT 是什么

Chain-of-Thought（思维链）通过中间步骤帮助模型完成多步问题。

```text
问题 -> 中间推导 -> 答案
```

常见方法：

| 方法 | 做法 | 适用场景 |
|:---|:---|:---|
| Zero-shot CoT | 提示模型分步求解 | 简单推理 |
| Few-shot CoT | 提供带步骤示例 | 固定题型 |
| Self-consistency | 多次采样后投票 | 可验证答案 |
| Tree of Thoughts | 搜索多个候选思路 | 组合搜索 |
| Program of Thoughts | 让程序执行计算 | 数学、数据处理 |

### 工程注意

不应把完整内部推理当成可靠解释或长期日志。生产系统更适合记录：

- `plan_summary`
- `decision`
- `evidence`
- `tool_call`
- `validation_result`

这些信息可审计，并且不依赖冗长自由文本。

## 2. 任务分解

一个好子任务应具备：

- 明确输入。
- 明确输出。
- 可独立验证。
- 依赖关系清晰。
- 粒度不过大也不过小。

示例目标：“比较三种向量数据库并给出选型建议”。

```json
[
  {
    "id": "collect_requirements",
    "output": "规模、延迟、部署、预算约束",
    "depends_on": []
  },
  {
    "id": "collect_evidence",
    "output": "候选数据库证据表",
    "depends_on": ["collect_requirements"]
  },
  {
    "id": "compare",
    "output": "按统一维度比较",
    "depends_on": ["collect_evidence"]
  },
  {
    "id": "recommend",
    "output": "结论、风险和迁移方案",
    "depends_on": ["compare"]
  }
]
```

## 3. Plan-and-Execute

Plan-and-Execute 将规划和执行分开：

```text
Planner -> Plan
Executor -> Execute next task
Evaluator -> Validate result
Replanner -> Update remaining plan
```

### 状态模型

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Task:
    id: str
    instruction: str
    status: str = "pending"
    result: Any = None
    error: str | None = None

@dataclass
class PlanState:
    goal: str
    tasks: list[Task] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    revision: int = 0
```

### 最小执行器

```python
def run_plan(state, execute, validate, max_replans=2):
    while True:
        pending = [t for t in state.tasks if t.status == "pending"]
        if not pending:
            return state

        task = pending[0]
        task.status = "running"

        try:
            task.result = execute(task, state.artifacts)
            verdict = validate(task, task.result)
            task.status = "completed" if verdict["ok"] else "failed"
            task.error = verdict.get("reason")
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)

        if task.status == "failed":
            if state.revision >= max_replans:
                return state
            state.tasks = replan(state)
            state.revision += 1
```

## 4. 什么时候重规划

不是每一步都要重新生成完整计划。适合重规划的信号：

- 工具返回的信息推翻关键假设。
- 必要资源不可用。
- 子任务验证失败。
- 用户修改目标或约束。
- 预算、时间或最大步数即将耗尽。

不应仅因为模型“觉得可以更好”就无限重规划。

## 5. DAG 规划

存在独立子任务时，用有向无环图表示依赖：

```text
             ┌-> 搜集 A ─┐
需求分析 ────┼-> 搜集 B ─┼-> 对比 -> 报告
             └-> 搜集 C ─┘
```

可并行的任务必须满足：

- 不写同一资源。
- 不依赖彼此结果。
- 失败可以独立重试。
- 合并规则明确。

## 6. Planner Prompt

```text
你是任务规划器。只负责拆分任务，不执行任务。

要求：
- 每个步骤必须产生可验证输出。
- 显式列出依赖。
- 不创建不必要步骤。
- 高风险动作标记 requires_approval=true。
- 最多生成 8 个步骤。

返回 JSON：
{
  "goal": "...",
  "assumptions": [],
  "tasks": [
    {
      "id": "...",
      "instruction": "...",
      "expected_output": "...",
      "depends_on": [],
      "requires_approval": false
    }
  ]
}
```

结构化输出比自然语言编号列表更适合执行。

## 7. 验证优先

规划系统的质量不取决于计划是否优美，而取决于每一步是否可验证。

| 任务类型 | 验证方式 |
|:---|:---|
| 代码修改 | 测试、lint、构建 |
| 数据查询 | schema、行数、约束检查 |
| 研究报告 | 引用存在性、证据覆盖率 |
| 网页操作 | URL、DOM 状态、截图 |
| 文件生成 | 文件存在、格式解析、视觉检查 |

## 8. 常见失败模式

### 计划过度

简单任务被拆成十几个步骤，成本和失败点增加。

修复：先判断是否需要计划；两步内能完成的任务直接执行。

### 计划漂移

执行过程中环境变化，但系统仍机械执行旧计划。

修复：在关键步骤后验证假设，满足重规划条件才更新计划。

### 子任务不可验证

例如“深入研究一下”。它没有清晰输出。

修复：“输出包含来源、发布日期、结论和限制的证据表”。

### Planner 和 Executor 相互污染

执行日志过多导致 Planner 忘记目标。

修复：Planner 只接收任务状态摘要、关键产物和失败原因。

### 反思循环

模型不断生成“可以进一步改进”。

修复：最大重规划次数、明确验收条件、预算上限。

## 9. ReAct 与 Plan-and-Execute

| 维度 | ReAct | Plan-and-Execute |
|:---|:---|:---|
| 决策粒度 | 每步即时决策 | 先形成全局计划 |
| 适合 | 短任务、探索任务 | 长任务、依赖明确 |
| 优势 | 灵活 | 可观察、可并行 |
| 风险 | 局部贪心、循环 | 初始计划过时 |

常见组合是：顶层用 Plan-and-Execute，每个子任务内部用有限步 ReAct。

## 10. 评测指标

- 最终任务成功率。
- 计划可执行率。
- 子任务一次通过率。
- 平均重规划次数。
- 无效步骤比例。
- 并行加速比。
- Token、时间和工具成本。

## 面试表达

> CoT 是模型形成中间推理的一类提示方法，Plan-and-Execute 是系统级任务控制模式。我会把计划表示成结构化任务 DAG，每步定义验收条件；执行失败时根据明确触发条件局部重规划，并限制重试和预算。日志记录可审计决策摘要、工具轨迹和验证结果，而不是依赖完整思维链。

## 练习

1. 把“分析一个 GitHub 项目”拆成最多 6 个可验证任务。
2. 为任务定义 JSON Schema。
3. 实现依赖满足后并行执行的调度器。
4. 加入最大成本和人工审批节点。

## 延伸阅读

- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [ReAct](https://arxiv.org/abs/2210.03629)
