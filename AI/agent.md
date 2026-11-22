

Agent = LLM + 记忆能力 + 思考能力(任务规划、分解、反思) + 工具使用






## memory

https://github.com/mem0ai/mem0

> 提供长期记忆的数据库和检索系统， 不同于短期记忆（LLM 上下文）

如何做到？：提取对话关键信息并存储，下次对话时检索并拼到 prompt

如何避免爆炸？：不存对话，只存结构化事实；检索时只取 top_k；

如何可解释（审计）？：有对记忆做归档和操作选项，memory.list()


长期来讲，感觉依赖 mem0 抽取事实不如写一份 profile 或规则文档。 mem0 优势处理一些动态的，不预先定义的信息，如“我昨天问了你什么 bug? 昨天任务到哪了？”。

最终形态：Profile + small memory



```python
import os
from mem0 import Memory

os.environ["DEEPSEEK_API_KEY"] = "sk-..."
# os.environ["OPENAI_API_KEY"] = "sk-..." # for embedder model

config = {
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": "deepseek-chat",  # default model
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0
        }
    },
    # "embedder": {
    #     "provider": "huggingface",
    #     "config": {
    #         "model": "BAAI/bge-small-en-v1.5"
    #     }
    # }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})
```



## reflection

很多任务不太需要，且容易引起连锁幻觉；

可选只在失败时触发








## other

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





