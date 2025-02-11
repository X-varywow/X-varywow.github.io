



Agent = LLM + 记忆能力 + 规划能力 + 工具使用能力


- 记忆能力
  - 短期记忆（上下文学习 prompt engineering）
  - 长期记忆（外部向量存储）
- 规划能力
  - 任务分解
  - 自我反思
- 工具使用能力


-----------

其它可能：多智能体协作

缺点：agent 的工作流程耗时会较长（推理耗时，交互耗时），空间认知、非现实认知不足; llm 本身的幻觉（误导信息）、可解释性问题任存在。

推理耗时：llm 在推理过程中采用自回归的序列生成方式，每个词的生成都依赖于之前生成的词。这种顺序依赖性限制了计算过程的并行化。


-----------

参考资料：
- [腾讯技术工程-AI Agent深度调研](https://mp.weixin.qq.com/s/smjNp8aX3nJrqw5-uZRQsQ)
- https://mp.weixin.qq.com/s/nHGv5C-xAvIFgOt_qWsL0g
- https://github.com/mem0ai/mem0
- https://docs.agpt.co/
- https://mp.weixin.qq.com/s/6Jn4-3KPoffsYGrrvYX6vg
- “记忆”的主流方案： https://arxiv.org/abs/2312.10997





