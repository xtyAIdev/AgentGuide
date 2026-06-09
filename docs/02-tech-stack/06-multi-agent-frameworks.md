# Multi-Agent 框架详解

> Multi-Agent 的价值不是让多个模型“开会”，而是把上下文、工具、权限和责任边界拆开。只有分工、并行或独立验证带来可测收益时，才值得增加多个 Agent。

## 1. 先判断是否真的需要

适合：

- 子任务可以并行。
- 不同角色需要不同工具或权限。
- 上下文太大，需要隔离。
- 需要独立 Reviewer。
- 不同任务适合不同模型。

不适合：

- 固定流程可用普通 workflow。
- 所有 Agent 使用相同上下文和工具。
- 没有评价协作收益的方法。
- 任务很短，通信成本大于执行成本。

## 2. 常见拓扑

### Supervisor

```text
             ┌-> Researcher
Supervisor ──┼-> Coder
             └-> Reviewer
```

Supervisor 分配任务并汇总结果。容易控制，但可能成为瓶颈。

### Pipeline

```text
Planner -> Researcher -> Writer -> Reviewer
```

适合步骤稳定的内容生产或数据处理。

### Handoff

当前 Agent 根据意图把控制权交给另一个 Agent：

```text
Triage -> Billing Agent
       -> Technical Support Agent
       -> Human Support
```

适合客服路由，但必须明确会话所有权和返回条件。

### Blackboard

多个 Agent 读写共享任务板：

```text
Shared Task Board
  ^      ^      ^
 A1     A2     A3
```

适合异步并行，但要处理冲突、重复任务和陈旧状态。

## 3. 框架对比

| 框架/模式 | 核心抽象 | 适合 | 注意点 |
|:---|:---|:---|:---|
| AutoGen AgentChat | Agent、Team、Message | 对话式多 Agent、研究原型 | 对话轮数和终止条件 |
| CrewAI | Agent、Task、Crew、Flow | 角色化业务流程 | 不要把角色描述当权限控制 |
| LangGraph | State、Node、Edge | 精确状态编排、恢复 | 需自己设计角色协议 |
| OpenAI Agents SDK | Agent、Tool、Handoff、Guardrail | 工具与 Handoff | 依赖具体模型生态 |
| 自研 Orchestrator | 自定义任务与队列 | 强控制、内部平台 | 维护成本和可观测性 |

框架 API 会变化，选型时重点看状态模型、恢复、追踪、审批和评测能力。

## 4. 任务协议

不要让 Agent 只发送自然语言“你帮我研究一下”。应使用任务信封：

```json
{
  "task_id": "research_001",
  "sender": "supervisor",
  "recipient": "researcher",
  "objective": "比较三种向量数据库",
  "inputs": {
    "requirements": ["10M vectors", "self-hosted"]
  },
  "expected_output": {
    "type": "evidence_table",
    "required_fields": ["claim", "source", "date", "limitation"]
  },
  "deadline": "2026-06-10T10:00:00Z",
  "budget": {"max_steps": 8},
  "permissions": ["web_read"]
}
```

## 5. 共享状态与私有上下文

建议分三层：

| 层 | 内容 |
|:---|:---|
| Shared State | 目标、任务状态、关键产物引用 |
| Agent Context | 角色规则、专属工具、当前子任务 |
| Trace | 消息、工具调用、错误、耗时 |

不要把所有 Agent 的完整对话互相广播，会造成成本增加和上下文污染。

## 6. Supervisor 最小实现

```python
def run_team(goal, agents, max_rounds=8):
    board = {
        "goal": goal,
        "tasks": [],
        "artifacts": {},
        "status": "running",
    }

    for _ in range(max_rounds):
        decision = supervisor_decide(board)

        if decision["type"] == "finish":
            board["status"] = "completed"
            return board

        if decision["type"] == "delegate":
            agent = agents[decision["recipient"]]
            result = agent.run(decision["task"], board["artifacts"])
            board["tasks"].append({
                "task": decision["task"],
                "agent": decision["recipient"],
                "result": result,
            })
            board["artifacts"].update(result.get("artifacts", {}))

    board["status"] = "max_rounds"
    return board
```

生产版本还需：

- 并发调度。
- 幂等任务。
- 超时与取消。
- 权限检查。
- 结果 schema。
- Checkpoint。

## 7. AutoGen 的设计思路

AutoGen 的 AgentChat 层强调 Agent 与 Team 的消息协作。使用时应关注：

- Agent 的工具和系统规则。
- Team 的发言选择策略。
- Termination Condition。
- 消息历史裁剪。
- 状态保存与恢复。

对话式团队适合探索，但生产流程最好把关键阶段固化为状态与验收条件。

## 8. CrewAI 的设计思路

CrewAI 常使用：

- Agent：角色、目标、工具。
- Task：任务与期望输出。
- Crew：多个 Agent 的协作集合。
- Flow：事件驱动、状态化业务流程。

角色描述只影响模型行为，不构成真实安全边界。权限必须在工具执行层实施。

## 9. Handoff 设计

Handoff 必须回答：

1. 谁拥有当前会话？
2. 转交哪些上下文？
3. 哪些敏感信息不能转交？
4. 子 Agent 完成后返回谁？
5. 转交失败怎么办？

路由结果建议结构化：

```json
{
  "target": "billing",
  "reason_summary": "用户询问重复扣款",
  "context": {"ticket_id": "T100"},
  "confidence": 0.93
}
```

## 10. 冲突解决

多个 Agent 给出不同结论时可使用：

- 证据优先：引用可验证来源。
- 规则优先：业务策略覆盖模型意见。
- Reviewer：按 rubric 独立评分。
- 人工仲裁：高风险或低置信度。

不要简单多数投票。同源模型可能产生高度相关的错误。

## 11. 安全设计

- 每个 Agent 只获得必要工具。
- 工具执行检查真实用户权限。
- 高风险操作需要审批。
- Agent 间消息视为不可信输入。
- 防止一个被注入的 Agent 控制整个团队。
- 限制最大轮数、成本和并发。

## 12. 评测 Multi-Agent

必须与单 Agent 基线比较：

| 指标 | 说明 |
|:---|:---|
| Task success | 最终成功率 |
| Coordination overhead | 协作消息和额外 Token |
| Delegation accuracy | 是否分给正确角色 |
| Merge quality | 汇总是否遗漏或冲突 |
| Parallel speedup | 并行是否真正降低时间 |
| Safety isolation | 权限隔离是否有效 |

如果成功率只提升 1%，成本却增长 4 倍，通常不值得。

## 13. 常见失败模式

- Agent 互相客套但不产生产物。
- Supervisor 重复委派同一任务。
- Reviewer 没有独立证据，只改写原答案。
- 共享上下文无限增长。
- 多个 Agent 同时修改同一文件。
- Handoff 后责任不清，任务无人结束。

修复核心是结构化任务、产物所有权、终止条件和可观测状态。

## 面试表达

> 我不会默认使用 Multi-Agent。先用单 Agent 或 workflow 建立基线；只有任务可并行、权限需隔离、上下文需拆分或需要独立审核时才引入多个角色。协作通过结构化任务信封和共享任务板完成，并用任务成功率、委派准确率、并行收益和协调成本评测。

## 延伸阅读

- [AutoGen AgentChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
