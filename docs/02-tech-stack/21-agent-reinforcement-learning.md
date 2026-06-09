# Agent 强化学习：从 SFT 到策略优化

> Agent 强化学习的目标不是让回答“更像人”，而是让策略在多步环境中更稳定地选择动作、调用工具并完成任务。它比普通文本偏好训练更难，因为奖励延迟、环境昂贵且轨迹具有随机性。

## 1. 先区分训练阶段

| 阶段 | 数据 | 目标 |
|:---|:---|:---|
| Pretraining | 大规模文本/代码 | 学习通用能力 |
| SFT | 指令与优质回答/轨迹 | 模仿目标行为 |
| Preference Optimization | chosen/rejected | 对齐偏好 |
| RL | 环境交互与奖励 | 优化多步策略 |

Agent 项目通常先做 Prompt 和系统工程，再考虑 SFT，最后才是 RL。

## 2. Agent 作为 MDP

$$
M=(S,A,P,R,\gamma)
$$

在工具型 Agent 中：

- $S$：任务、历史、工具观察、环境状态。
- $A$：文本、工具名和参数、停止。
- $P$：工具或环境执行后的状态变化。
- $R$：成功、过程、安全和成本奖励。
- $\pi_\theta(a|s)$：模型策略。

与游戏不同，Agent 的动作空间通常是巨大文本空间，因此需要结构化 Tool Calling 降低难度。

## 3. 为什么先做 SFT

如果模型连基本工具格式都不会，直接 RL 会浪费大量探索。

SFT 数据可来自：

- 人工高质量轨迹。
- 规则系统生成。
- 强模型生成后人工筛选。
- 成功线上轨迹脱敏。
- 失败轨迹修正。

轨迹样例：

```json
{
  "task": "查询订单状态，不要修改订单",
  "steps": [
    {
      "state_summary": "已获得订单号 A100",
      "action": {
        "tool": "get_order",
        "arguments": {"order_id": "A100"}
      },
      "observation": {"status": "shipped"}
    }
  ],
  "final": "订单已发货。",
  "success": true
}
```

不要训练模型模仿冗长、未经验证的思维链。

## 4. Reward 设计

总奖励可以写成：

$$
R
=
w_sR_{success}
+w_pR_{process}
+w_fR_{format}
-w_cC_{cost}
-w_vP_{violation}
$$

### Outcome Reward

- 任务是否完成。
- 测试是否通过。
- 最终环境状态是否正确。

### Process Reward

- 是否选择必要工具。
- 参数是否正确。
- 是否引用证据。
- 是否恢复错误。

### Safety Penalty

- 越权。
- 泄漏敏感数据。
- 执行禁止动作。

### Cost Penalty

- Token。
- 工具调用。
- 延迟。
- 环境资源。

奖励设计不当会产生 Reward Hacking。例如只奖励“少调用工具”，模型可能直接猜答案。

## 5. PPO

PPO 是经典 on-policy 策略优化方法。核心使用概率比率：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
$$

裁剪目标：

$$
L^{CLIP}
=
\mathbb{E}
\left[
\min(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
)
\right]
$$

它限制单次更新幅度，降低策略崩坏风险。

PPO 通常需要：

- Policy Model。
- Reference Model。
- Value/Critic。
- Reward Model 或环境奖励。
- 在线 Rollout。

优点是可直接优化序列级奖励，缺点是复杂、显存和采样成本高。

## 6. DPO

DPO 使用偏好对：

```text
prompt, chosen, rejected
```

它不需要显式训练 Reward Model 和在线 Rollout，训练更简单。DPO 更适合回答级偏好或固定轨迹偏好，但不等价于真实环境中的多步 RL。

对于 Agent，可构造：

- 成功轨迹 vs 失败轨迹。
- 安全轨迹 vs 越权轨迹。
- 高效轨迹 vs 冗余轨迹。

必须控制长度偏差，否则模型可能仅偏好更长或更短输出。

## 7. GRPO

GRPO 类方法对同一问题采样一组候选，根据组内相对奖励估计优势，减少对独立 Value Model 的依赖。

直觉：

```text
同一任务生成 G 条轨迹
 -> 环境或评分器打分
 -> 用组内均值/方差标准化
 -> 提升高于组平均的轨迹概率
```

适合存在可验证奖励的数学、代码或工具任务。组内样本过于相似、奖励稀疏或评分器不可靠时，训练效果会受限。

## 8. Agent Rollout

```python
def rollout(env, policy, max_steps=12):
    state = env.reset()
    trajectory = []

    for step in range(max_steps):
        action = policy.sample(state)
        next_state, reward, done, info = env.step(action)
        trajectory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "info": info,
        })
        state = next_state
        if done:
            break

    return trajectory
```

生产训练环境必须可重置、可复现且隔离，不能让训练 Agent 直接操作真实生产系统。

## 9. Credit Assignment

最终失败可能源于第 2 步错误工具，但奖励直到第 10 步才出现。

解决思路：

- 过程奖励。
- 子目标奖励。
- Verifier 检查每步。
- 从失败点回放。
- 训练 Value/Process Reward Model。

过程奖励必须谨慎，过强会限制模型探索替代路径。

## 10. 环境构建

好的 Agent 训练环境应具备：

- 确定性 Fixture。
- 快速 Reset。
- Tool schema 固定。
- 动作日志。
- 超时和最大步数。
- 网络和文件隔离。
- 自动评分。

例如 Coding Agent 使用容器：

```text
base repository snapshot
 -> apply agent action
 -> run tests
 -> reward
 -> destroy container
```

## 11. 数据管线

```text
Tasks
 -> Rollout
 -> Trace Validation
 -> Reward
 -> Filter/Deduplicate
 -> Train
 -> Offline Eval
 -> Canary Eval
```

记录：

- Task 版本。
- Environment 版本。
- Tool 版本。
- Model checkpoint。
- Sampling 参数。
- Reward 组件。

否则结果不可复现。

## 12. 防止 Reward Hacking

- 隐藏测试与公开测试分开。
- Outcome 与过程奖励组合。
- 随机化环境表面特征。
- 审计高奖励异常轨迹。
- 多评分器交叉验证。
- 保留安全硬约束，不把一切交给奖励。

## 13. 训练前的决策清单

只有满足以下条件才考虑 RL：

- 已有稳定环境和自动评分。
- Prompt/SFT 已达到瓶颈。
- 有足够任务和计算预算。
- 改进可通过固定 Eval 衡量。
- 安全隔离和数据治理完备。

否则优先：

- 改 Tool。
- 改 Context。
- 改 Workflow。
- 增加 SFT 轨迹。
- 使用更合适模型。

## 14. 评测

- Task success。
- pass^k。
- 平均步数。
- Tool error rate。
- Safety violation。
- Reward 与真实业务指标相关性。
- 与 SFT-only 基线比较。

必须检查训练集外任务和环境扰动。

## 面试表达

> Agent RL 把模型视为多步策略，状态包含任务和工具观察，动作是结构化工具调用，奖励结合任务成功、过程、安全和成本。我会先用 SFT 建立基本行为，再在可重置、可自动评分的沙盒中采样轨迹。PPO 适合在线策略优化，DPO 更像离线偏好优化，GRPO 利用组内相对奖励降低 Value Model 依赖。训练后用独立任务、pass^k 和安全指标验证。

## 延伸阅读

- [PPO](https://arxiv.org/abs/1707.06347)
- [DPO](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [Agent Lightning](https://github.com/microsoft/agent-lightning)
