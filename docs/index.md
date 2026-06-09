# AgentGuide 系统学习手册

本站严格按照项目根目录 `README.md` 中的“第五步：系统学习 Agent 技术”组织内容。你可以从左侧目录按模块连续学习，也可以使用顶部搜索快速定位知识点。

## 学习顺序

### L1 基础认知层

1. **模块1：Agent 核心概念解析**
   - [什么是 AI Agent？](01-theory/01-what-is-agent.md)
2. **模块2：技术演进历程与趋势洞察**
   - [Agent 技术演进史](01-theory/02-agent-history.md)
3. **模块3：大模型工作原理**
   - [Transformer 架构详解](01-theory/03-transformer.md)
   - [DeepSeek 系列完整深度笔记](01-theory/10-deepseek-series.md)
   - [LLaMA 系列完整深度笔记](01-theory/11-llama-series.md)
   - [Qwen 系列深度学习笔记](01-theory/12-qwen-series.md)

### L2 开发实现层

4. **模块4：经典 Agent 范式手撕实现**
   - [手撕 ReAct](01-theory/04-react-framework.md)
   - [规划与执行](01-theory/05-cot-and-planning.md)
5. **模块5：低代码平台快速验证**
   - [框架对比与 LangChain 指南](02-tech-stack/04-langchain-guide.md)
   - [Multi-Agent 框架详解](02-tech-stack/06-multi-agent-frameworks.md)
6. **模块6：主流框架深度实战**
   - [LangGraph 完整教程](02-tech-stack/04-langchain-guide.md)
   - [AutoGen 实战指南](02-tech-stack/06-multi-agent-frameworks.md)
   - [AgentScope 快速上手](02-tech-stack/07-agentscope.md)
   - [CrewAI 企业实战](02-tech-stack/06-multi-agent-frameworks.md)
7. **模块7：自研 Agent 框架设计原理**
   - [打造自己的 Agent 框架](02-tech-stack/22-build-your-agent-framework.md)

### L3 高阶优化层

8. **模块8：检索增强生成（RAG）全栈技术**
   - [RAG 系统开发指南](02-tech-stack/20-rag-full-pipeline.md)
   - [向量数据库选型](02-tech-stack/08-vector-db-basics.md)
9. **模块9：上下文工程**
   - [上下文工程资源合集](02-tech-stack/13-context-engineering-resources.md)
   - [Context Engineering 2.0](02-tech-stack/18-context-engineering-guide.md)
10. **模块10：智能体通信标准与协议**
    - [MCP 完全指南](02-tech-stack/14-mcp-protocol.md)
11. **模块11：模型微调与强化学习**
    - [Agent 强化学习](02-tech-stack/21-agent-reinforcement-learning.md)
    - [SFT 监督微调](02-tech-stack/16-sft-finetuning.md)
    - [Post-Training 完整指南](02-tech-stack/25-post-training-complete-guide.md)
12. **模块12：性能评估与效果量化**
    - [科学评估 Agent](01-theory/09-evaluation-metrics.md)
    - [AgentBench 详解](01-theory/08-agent-bench.md)
    - [AI Agent 评估完全指南](02-tech-stack/agent-evaluation-complete-guide.md)

### 工程化与项目

13. **生产级系统设计**
    - [高可用 RAG 系统](03-practice/02-high-availability-rag.md)
    - [Agent 安全性指南](03-practice/03-agent-security.md)
14. **简历级实战项目**
    - [毕业设计完整指南](03-practice/04-graduation-project.md)

## 本地阅读

在项目根目录执行：

```powershell
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

然后访问 <http://127.0.0.1:8000>。修改 Markdown 后，页面会自动刷新。
