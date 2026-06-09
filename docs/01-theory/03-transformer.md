# Transformer 架构详解

> Transformer 是现代大语言模型的基础。理解它不需要先掌握所有数学细节，但必须弄清 Token、Embedding、Attention、位置编码、前馈网络和生成过程如何连接。

## 1. 整体结构

输入文本首先被切成 Token，再映射为向量：

```text
文本
 -> Tokenizer
 -> Token IDs
 -> Token Embedding + Position Information
 -> N 个 Transformer Block
 -> Linear + Softmax
 -> 下一个 Token 概率
```

一个典型 Decoder-only Block：

```text
x
├─ RMSNorm/LayerNorm
├─ Causal Self-Attention
├─ Residual Add
├─ RMSNorm/LayerNorm
├─ MLP 或 MoE
└─ Residual Add
```

GPT、LLaMA、Qwen、DeepSeek 等生成模型主要采用 Decoder-only 架构。

## 2. Tokenizer：模型看到的不是“字”

Tokenizer 将字符串转成离散 ID：

```python
text = "Agent 会调用工具"
token_ids = tokenizer.encode(text)
```

常见方法包括 BPE、WordPiece 和 SentencePiece。一个 Token 可能是汉字、词片段、标点或代码符号。

### 为什么 Agent 开发要关心 Token

- 上下文窗口按 Token 而不是字符计费。
- 工具返回过长会挤掉关键指令。
- JSON、代码和中文的 Token 密度不同。
- 截断可能破坏结构化数据。

## 3. Embedding：把离散 ID 变成向量

词表大小为 $V$，隐藏维度为 $d$，Embedding 矩阵为：

$$
E \in \mathbb{R}^{V \times d}
$$

Token ID $i$ 对应矩阵第 $i$ 行。经过 Embedding 后，长度为 $n$ 的输入变为：

$$
X \in \mathbb{R}^{n \times d}
$$

向量本身不直接包含顺序，因此还需要位置编码。

## 4. Self-Attention

输入 $X$ 通过三个线性变换得到：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

- $Q$：当前位置想查什么。
- $K$：每个位置提供什么索引。
- $V$：每个位置真正携带的信息。
- $M$：Mask；生成模型用它阻止看到未来 Token。

### 直觉示例

句子“张三把书交给李四，因为他要离开”中，“他”应关注谁。Attention 会根据上下文计算“他”与其他 Token 的相关权重，再汇总对应信息。

### 最小 NumPy 实现

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)

def attention(q, k, v, mask=None):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ v, weights
```

## 5. Causal Mask

自回归模型预测第 $t$ 个 Token 时只能看到此前内容。

```text
      k1 k2 k3 k4
q1     ✓  ×  ×  ×
q2     ✓  ✓  ×  ×
q3     ✓  ✓  ✓  ×
q4     ✓  ✓  ✓  ✓
```

若训练时泄漏未来 Token，模型在推理时就无法复现训练条件。

## 6. Multi-Head Attention

单个注意力头只能在一个投影空间中计算关系。多头注意力并行使用多组参数：

$$
\operatorname{MHA}(X)
=
\operatorname{Concat}(head_1,\ldots,head_h)W_O
$$

不同头可能分别关注：

- 局部语法关系。
- 长距离指代。
- 特定分隔符。
- 代码括号配对。
- 工具调用结构。

不要把某个头机械解释成固定语义。模型内部表示是分布式的。

## 7. 位置编码与 RoPE

### 绝对位置编码

原始 Transformer 使用正弦与余弦函数：

$$
PE_{(pos,2i)}=\sin(pos/10000^{2i/d})
$$

$$
PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d})
$$

### RoPE

现代开源 LLM 常使用旋转位置编码。RoPE 对 Query 和 Key 的二维子空间进行与位置相关的旋转，使注意力分数自然包含相对位置信息。

工程上要记住：

- 扩展上下文窗口不只是修改一个数字。
- 超出训练分布可能导致长上下文性能下降。
- “支持 1M Token”不等于能稳定利用任意位置的信息。

## 8. Feed-Forward Network

Attention 负责 Token 间的信息混合，FFN 对每个位置独立做非线性变换：

$$
\operatorname{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2
$$

现代模型常用 SwiGLU：

$$
\operatorname{SwiGLU}(x)
=
(\operatorname{SiLU}(xW_g)\odot xW_u)W_d
$$

FFN 参数通常占模型参数的大部分。

## 9. Residual 与 Normalization

残差连接：

$$
y=x+F(x)
$$

它帮助深层网络传递梯度和保留原始信息。LayerNorm 或 RMSNorm 用于稳定激活尺度。

很多现代 LLM 使用 Pre-Norm：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

## 10. MoE：不是每个 Token 都用全部参数

Mixture of Experts 将 FFN 替换为多个专家，并由 Router 为每个 Token 选择少量专家：

```text
Token
 -> Router
 -> Top-k Experts
 -> Weighted Merge
```

优势：

- 总参数量可以很大。
- 单 Token 只激活部分参数，计算量相对可控。

挑战：

- 负载均衡。
- 跨设备通信。
- Expert capacity。
- 路由稳定性。

## 11. 训练目标

自回归语言模型最常见目标是预测下一个 Token：

$$
\mathcal{L}
=
-\sum_{t=1}^{n}\log p_\theta(x_t|x_{<t})
$$

训练时通常一次并行计算所有位置的损失；推理时则逐 Token 生成。

## 12. 推理与 KV Cache

生成第一个 Token 时，模型计算整段 Prompt 的 Key 和 Value。之后无需重复计算历史 Token，可把它们缓存：

```text
Prefill: 处理完整输入，计算 KV Cache
Decode: 每次加入一个新 Token，复用历史 KV
```

KV Cache 的显存开销大致随以下因素线性增长：

- 层数。
- 序列长度。
- KV 头数。
- 每头维度。
- 并发请求数。

这也是上下文越长、并发越贵的重要原因。

## 13. Sampling

模型输出 logits，经过 Softmax 得到概率。常见采样参数：

| 参数 | 作用 |
|:---|:---|
| temperature | 调整概率分布尖锐程度 |
| top-k | 只保留概率最高的 k 个 Token |
| top-p | 保留累计概率达到 p 的 Token |
| repetition penalty | 降低重复概率 |

高可靠工具调用通常使用较低温度，但温度低不保证事实正确。

## 14. Transformer 与 Agent 的关系

理解 Transformer 能解释很多 Agent 问题：

| Agent 现象 | 底层原因 |
|:---|:---|
| 长上下文仍漏信息 | 注意力不是数据库精确查询 |
| 工具结果过长导致指令失效 | 有限窗口与注意力竞争 |
| 输出存在随机性 | 自回归概率采样 |
| 多轮成本逐渐升高 | Prompt 与 KV Cache 增长 |
| 中间信息被忽略 | 位置与显著性影响 |

因此 Agent 需要 Context Engineering，而不是无限堆上下文。

## 15. 面试高频问题

### 为什么除以 $\sqrt{d_k}$？

维度增大时点积方差会变大，Softmax 容易饱和。缩放使数值和梯度更稳定。

### Encoder-only、Decoder-only、Encoder-Decoder 有何区别？

- Encoder-only：双向理解，常用于分类和表示。
- Decoder-only：因果生成，主流 LLM 架构。
- Encoder-Decoder：输入编码后由 Decoder 生成，常用于翻译和摘要。

### Attention 的复杂度是什么？

标准全注意力的时间和注意力矩阵空间复杂度约为 $O(n^2)$。实际推理还需考虑 KV Cache、带宽和批处理。

## 练习

1. 手算一个 3 Token、2 维向量的 Attention。
2. 修改 NumPy 示例加入 causal mask。
3. 比较相同文本在不同 Tokenizer 下的 Token 数。
4. 解释为什么 RAG 不能简单替换长上下文。

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
