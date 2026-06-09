# 高可用 RAG 系统实战

> 高可用 RAG 的目标不是“永不失败”，而是在模型、向量库、数据源或外部 API 出问题时，系统能够限时失败、可观测、可降级、可恢复，并且不返回危险的伪答案。

## 1. 生产架构

```text
Client
 -> API Gateway
 -> Auth / Rate Limit
 -> RAG Orchestrator
    -> Query Service
    -> Retrieval Service
       -> Keyword Index
       -> Vector DB
       -> Reranker
    -> Context Service
    -> Model Gateway
 -> Citation Validator
 -> Response

Control Plane:
Ingestion / Index Version / Eval / Observability / Admin
```

在线查询链路和离线索引链路应分离，避免批量导入拖垮在线服务。

## 2. SLO

示例目标：

| 指标 | SLO |
|:---|:---|
| 可用性 | 99.9% |
| P95 延迟 | < 4 秒 |
| 检索成功率 | > 99.5% |
| 引用有效率 | > 99% |
| 越权检索 | 0 |

错误预算：

$$
ErrorBudget = 1 - SLO
$$

SLO 应区分：

- 系统可用。
- 答案质量。
- 安全正确。

HTTP 200 但答案错误并不算真正成功。

## 3. 超时预算

总请求 4 秒，可分配：

```text
Auth              50 ms
Query rewrite    300 ms
Retrieval        500 ms
Rerank           500 ms
Generation      2200 ms
Validation       300 ms
Buffer           150 ms
```

每层必须有独立超时，不能只设置入口总超时。

## 4. 重试

只对暂时性错误重试：

- 429。
- 连接重置。
- 短暂 5xx。

不重试：

- 400 参数错误。
- 401/403 权限错误。
- 内容安全拒绝。
- 确定性 schema 错误。

指数退避加随机抖动：

```python
import random
import time

def backoff(attempt, base=0.2, cap=2.0):
    delay = min(cap, base * 2 ** attempt)
    time.sleep(random.uniform(0, delay))
```

重试必须受总超时预算约束。

## 5. Circuit Breaker

当依赖持续失败：

```text
CLOSED -> 正常请求
失败超过阈值
OPEN -> 快速失败/降级
冷却时间后
HALF_OPEN -> 少量探测
成功 -> CLOSED
失败 -> OPEN
```

避免每个请求都等待已经故障的服务。

## 6. 降级策略

| 故障 | 降级 |
|:---|:---|
| Reranker 不可用 | 使用融合检索排序 |
| Vector DB 故障 | 关键词检索 |
| Query Rewrite 故障 | 使用原 Query |
| 主模型故障 | 备用模型或返回检索摘要 |
| 全部检索故障 | 明确告知暂不可查询 |
| Citation 验证失败 | 返回“证据不足”，不强答 |

降级结果必须标记，便于监控和用户理解。

## 7. 缓存

### Embedding Cache

Key 包含：

```text
hash(text) + embedding_model_version
```

### Retrieval Cache

必须包含：

```text
normalized_query
+ tenant/user permission scope
+ index_version
+ filters
```

忽略权限会造成数据泄漏。

### Answer Cache

只适合：

- 稳定公开知识。
- 非个性化查询。
- 可接受短时间陈旧。

缓存条目应保留来源和生成版本。

## 8. 限流与背压

分层限流：

- 用户。
- 租户。
- API Key。
- 模型。
- 检索服务。

当队列积压时：

- 拒绝低优先级请求。
- 降低并行检索。
- 使用轻量模型。
- 停止接受批量任务。

不能无限排队，否则超时请求仍消耗资源。

## 9. Bulkhead

将资源池隔离：

- 在线问答。
- 批量索引。
- 管理后台。
- Eval 任务。
- 高级 Agent 查询。

一个大客户或批处理任务不应占满所有连接和线程。

## 10. 数据与索引高可用

### 版本化

```text
documents_v12
chunks_v12
embeddings_model_x_v12
index_v12
```

### 蓝绿发布

1. 构建新索引。
2. 检查数量、维度和分布。
3. 跑固定 Eval。
4. 小流量 Canary。
5. 切换 Alias。
6. 保留旧索引用于回滚。

### 备份与恢复

定义：

- RPO：最多允许丢多少数据。
- RTO：多久恢复。

定期实际演练恢复，不能只确认“备份任务成功”。

## 11. 幂等 Ingestion

```text
document_id + source_version + parser_version
+ chunker_version + embedding_version
```

同一批次重复执行不应产生重复 Chunk。

导入状态：

```text
RECEIVED -> PARSED -> CHUNKED -> EMBEDDED
 -> INDEXED -> VALIDATED -> PUBLISHED
```

失败后从最近成功阶段恢复。

## 12. Model Gateway

统一处理：

- 模型路由。
- 超时和重试。
- 费用预算。
- 限流。
- Fallback。
- Prompt/模型版本。
- 输出 schema。
- 内容安全。

不要让每个业务服务直接实现一套模型调用逻辑。

## 13. 可观测性

### Trace

```text
request_id
 -> rewrite span
 -> retrieval spans
 -> rerank span
 -> model span
 -> validation span
```

### Metrics

系统：

- QPS。
- P50/P95/P99。
- 错误率。
- 超时率。
- 熔断状态。
- 队列长度。

质量：

- No-answer rate。
- Citation validity。
- Retrieval hit rate。
- 用户纠错率。
- 降级比例。

成本：

- Token/request。
- Embedding cost。
- Rerank cost。
- Cache hit rate。

### Logs

日志结构化并脱敏。不要记录完整用户隐私、密钥或未经处理的内部文档。

## 14. 健康检查

- Liveness：进程是否活着。
- Readiness：是否能接收流量。
- Dependency health：向量库、模型、缓存是否可用。
- Synthetic query：完整 RAG 链路是否能回答固定问题。

仅检查 `/health = 200` 无法证明 RAG 可用。

## 15. 安全

- 认证与租户隔离。
- 查询阶段 ACL Filter。
- 工具与数据最小权限。
- 检索内容视为不可信。
- Prompt Injection 防护。
- 引用与输出脱敏。
- 删除请求覆盖索引、缓存和日志。

安全策略优先于可用性降级：不能为了“系统可用”而绕过权限。

## 16. 灰度发布

灰度对象：

- 新 Embedding。
- 新 Chunk 策略。
- 新 Reranker。
- 新 Prompt。
- 新模型。

分流时固定用户或会话，避免同一会话在不同版本间跳动。

比较：

- 质量指标。
- 延迟。
- 成本。
- 错误率。
- 安全指标。

## 17. 故障演练

定期注入：

- 向量库超时。
- 模型 429。
- Reranker 500。
- Redis 不可用。
- 新索引为空。
- 权限服务延迟。

验证：

- 是否按预期降级。
- 告警是否触发。
- Runbook 是否可执行。
- 是否产生越权或伪答案。

## 18. Runbook 示例

### Vector DB P95 突增

1. 检查 QPS 与连接池。
2. 检查索引加载和内存。
3. 检查 Metadata Filter 是否退化。
4. 降低候选数或切关键词检索。
5. 开启熔断。
6. 保存样本请求用于复盘。

### 引用错误率升高

1. 对比最近 Prompt/模型/Rerank 变更。
2. 检查索引版本。
3. 回滚到上一个稳定版本。
4. 把失败案例加入 Eval。

## 19. 上线清单

- [ ] 明确 SLO 和错误预算。
- [ ] 每个依赖有超时。
- [ ] 重试区分错误类型。
- [ ] 有熔断、限流和背压。
- [ ] 有明确降级策略。
- [ ] 索引可回滚。
- [ ] 权限过滤前置。
- [ ] Trace、Metrics、Logs 完整。
- [ ] 有固定 Eval 和 Canary。
- [ ] 做过备份恢复与故障演练。

## 面试表达

> 高可用 RAG 要同时保障服务、质量和安全。我会拆分在线与离线链路，设置分层超时、有限重试、熔断、限流、资源隔离和可解释降级；索引用版本化和蓝绿发布，缓存键包含权限与索引版本。通过完整 Trace、质量指标、合成查询和故障演练验证系统，而不是只看 HTTP 可用率。

## 延伸阅读

- [AWS Builders Library: Timeouts, retries and backoff](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
