

Agent = LLM + 记忆能力 + 思考能力(任务规划、分解、反思) + 工具使用










-----------

其它可能：多智能体协作

缺点：agent 的工作流程耗时会较长（推理耗时，交互耗时），空间认知、非现实认知不足; llm 本身的幻觉（误导信息）、可解释性问题任存在。

推理耗时：llm 在推理过程中采用自回归的序列生成方式，每个词的生成都依赖于之前生成的词。这种顺序依赖性限制了计算过程的并行化。


-----------

参考资料：
- [Hello-Agents](https://github.com/datawhalechina/hello-agents)⭐️
- [腾讯技术工程-AI Agent深度调研](https://mp.weixin.qq.com/s/smjNp8aX3nJrqw5-uZRQsQ)
- https://mp.weixin.qq.com/s/nHGv5C-xAvIFgOt_qWsL0g
- https://github.com/mem0ai/mem0
- https://docs.agpt.co/
- “记忆”的主流方案： https://arxiv.org/abs/2312.10997





