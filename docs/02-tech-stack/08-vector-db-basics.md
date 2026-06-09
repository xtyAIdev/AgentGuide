# 向量数据库基础与选型

> 向量数据库解决“在大量向量中找到相似对象”的问题。它是 RAG 的索引层，不负责文档解析、答案生成，也不能自动保证检索相关性。

## 1. Embedding 是什么

Embedding 模型将对象映射到向量：

$$
f(x) \rightarrow \mathbf{v}\in\mathbb{R}^{d}
$$

语义相近对象通常在向量空间中距离更近。

常见对象：

- 文本段落。
- 图片。
- 音频。
- 商品。
- 用户行为。
- 代码。

Embedding 模型、维度和归一化方式必须写入索引元数据。不能在同一索引中无计划地混用不同模型向量。

## 2. 相似度

### Cosine Similarity

$$
\cos(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x}\cdot\mathbf{y}}
{\|\mathbf{x}\|\|\mathbf{y}\|}
$$

关注方向，文本语义检索常用。

### Inner Product

$$
\operatorname{IP}(\mathbf{x},\mathbf{y})=\mathbf{x}\cdot\mathbf{y}
$$

向量归一化后，Inner Product 与 Cosine 排序通常等价。

### Euclidean Distance

$$
L2(\mathbf{x},\mathbf{y})
=
\sqrt{\sum_i(x_i-y_i)^2}
$$

选择距离函数应遵循 Embedding 模型说明，不能凭感觉切换。

## 3. 精确检索与 ANN

精确检索对每个向量计算距离，复杂度近似：

$$
O(Nd)
$$

数据规模大时通常使用 Approximate Nearest Neighbor：

- 牺牲少量召回率。
- 显著降低延迟。
- 需要调索引和搜索参数。

## 4. 主流索引

### HNSW

Hierarchical Navigable Small World 构建多层近邻图。

特点：

- 低延迟、高召回。
- 在线查询表现好。
- 内存占用较高。
- 构建和更新成本需评估。

常见参数：

- `M`：每个节点连接数量。
- `efConstruction`：构建时搜索范围。
- `efSearch`：查询时搜索范围。

`efSearch` 越大，召回通常越高，延迟也越高。

### IVF

先训练聚类中心，把向量分桶。查询时只搜索部分桶。

常见参数：

- `nlist`：桶数量。
- `nprobe`：查询桶数量。

适合大规模数据；需要训练索引并调节召回与延迟。

### Product Quantization

将向量分段量化，减少内存和存储。适合超大规模，但精度会下降。

## 5. 一条记录应存什么

```json
{
  "id": "doc-10#chunk-3",
  "vector": [0.01, -0.12, 0.33],
  "text": "退款申请应在签收后十四天内提交。",
  "metadata": {
    "doc_id": "doc-10",
    "title": "退款政策",
    "section": "申请期限",
    "tenant_id": "tenant-a",
    "language": "zh",
    "updated_at": "2026-05-01",
    "embedding_model": "model-version"
  }
}
```

至少保留：

- 稳定 chunk id。
- 原文或原文引用。
- 文档与章节信息。
- 权限/租户字段。
- 更新时间。
- Embedding 版本。

## 6. Metadata Filter

向量相似不等于用户有权读取。检索必须同时满足：

```text
semantic_similarity
AND tenant_id = current_tenant
AND access_level <= user_level
AND status = published
```

权限过滤应尽量在数据库查询阶段执行，而不是取回后再过滤。

## 7. Hybrid Search

向量检索擅长语义，关键词检索擅长：

- 产品编号。
- 人名。
- 错误码。
- 专有名词。
- 精确短语。

混合检索可融合 BM25 与 Vector Search。

### Reciprocal Rank Fusion

$$
RRF(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}
$$

它只依赖排名，适合不同分数尺度的结果融合。

```python
def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

## 8. Reranker

第一阶段检索追求召回，第二阶段 Reranker 对 Query-Document 对进行更精细打分：

```text
Top 100 ANN/BM25
 -> Reranker
 -> Top 5-10 Context
```

Rerank 增加延迟和费用，应在离线评测中验证收益。

## 9. Milvus、Qdrant、Chroma 怎么选

| 维度 | Milvus | Qdrant | Chroma |
|:---|:---|:---|:---|
| 定位 | 分布式向量数据库 | 向量搜索引擎/数据库 | 轻量 Embedding 数据库 |
| 适合 | 大规模、复杂部署 | 中大型应用、过滤与工程体验 | 本地原型、教学 |
| 运维 | 较复杂 | 中等 | 简单 |
| 扩展性 | 强 | 强 | 以轻量场景为主 |
| 生产建议 | 先做容量与运维评估 | 适合自托管常规项目 | 原型优先，生产需压测 |

还可以评估托管服务、PostgreSQL + pgvector 或云厂商搜索服务。不存在适合所有项目的唯一答案。

## 10. 选型问题

先收集：

1. 向量数量和增长速度。
2. 向量维度。
3. QPS、P95/P99 延迟。
4. 更新和删除频率。
5. Metadata Filter 复杂度。
6. 多租户隔离要求。
7. 是否需要混合检索。
8. 备份、容灾和合规要求。
9. 团队运维能力。
10. 预算。

## 11. 容量估算

仅原始 float32 向量存储约为：

$$
Storage=N\times d\times 4\ bytes
$$

一千万条、1024 维：

$$
10^7\times1024\times4
\approx40.96\ GB
$$

这还不包括：

- 索引开销。
- Metadata。
- 副本。
- WAL。
- 临时构建空间。

## 12. 写入管线

```text
Document
 -> Parse
 -> Normalize
 -> Chunk
 -> Embed
 -> Validate dimension
 -> Upsert
 -> Verify count
 -> Mark index version ready
```

使用确定性 ID，确保重复导入是幂等的。

## 13. 更新与删除

- 文档更新：生成新版本，验证后切换别名。
- Embedding 升级：建立新索引，不直接覆盖旧索引。
- 删除请求：删除向量、原文、缓存和派生产物。
- 失败恢复：记录导入批次和状态。

蓝绿索引：

```text
index_v1 (serving)
index_v2 (building)
 -> offline eval
 -> switch alias
 -> retain v1 for rollback
```

## 14. 评测

### 检索质量

- Recall@k。
- Precision@k。
- MRR。
- nDCG。
- Context precision。

### 系统指标

- P50/P95/P99。
- QPS。
- Index build time。
- Upsert latency。
- Memory/disk。
- 错误率。

必须用真实查询集，不要只用随机向量压测。

## 15. 常见误区

- 相似度分数不能跨模型直接比较。
- Top-k 越大不一定越好，会增加噪声。
- Chunk 越小不一定越准，会丢上下文。
- 向量库不是权限系统。
- ANN 参数默认值不一定适合你的数据。
- 更换 Embedding 模型必须重建索引。

## 面试表达

> 我会根据数据规模、过滤条件、更新频率、延迟和运维能力选向量库。检索使用语义、关键词和 Rerank 的分层架构；权限过滤前置，索引带模型版本并用蓝绿方式升级。效果用 Recall@k、MRR 和端到端答案指标评估，系统侧压测 P95/P99、QPS 和容量。

## 延伸阅读

- [Milvus Documentation](https://milvus.io/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [pgvector](https://github.com/pgvector/pgvector)
