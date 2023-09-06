
## ChatGPT

ChatGPT is a sibling model to InstructGPT.

Generative Pre-trained Transformer

可参考 instruct gpt 来学习，[论文地址](https://arxiv.org/abs/2203.02155)

学习阶段：
- 学习文字接龙
- 人类老师引导文字接龙方向
- 模仿人类老师的喜好
- 用增强式学习向模拟老师学习

增强式学习（点赞或踩），相对于督导式（给确定答案），更加省时省力


预训练（自督导式学习），基石模型

微调（finetune）

---------------

挑战部分（参考 chatglm ppt）
- 训练成本高昂, 1750 亿参数的 GPT3 使用了上万块 V100，时机费 460w 美元，总成本预估 1200w 美元
- 人力投入极大，google PaLM 540B 团队，前期准备29人，训练过程11人，作者列表68人
- 训练过程不稳定，容易出现训练不收敛现象





</br>

_相关工作_

**（1）Prompting**

精准提出需求，参考文字接龙。让 ChatGPT 偏向应用场景的话，先对其进行催眠。

eg：请想象你是我的朋友，我会对你抱怨，希望你可以用中文提供安慰，并试图跟我聊聊，在对话过程中展现同理心，现在我们开始。

[ChatGPT 中文调教指南](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)

[提示工程指南](https://www.promptingguide.ai/zh)

**（2）Auto-GPT**

让 GPT 自动完成任务，Prompt 工程的一个前沿。（2023.04.17 80k star）

https://github.com/Significant-Gravitas/Auto-GPT

**（3）AI 平民化**

[ColossalAI](https://github.com/hpcaitech/ColossalAI) , Making large AI models cheaper, faster and more accessible

[DeepSpeed 官网](https://www.deepspeed.ai/)

[教程一：DeepSpeed Chat: 一键式RLHF训练](https://zhuanlan.zhihu.com/p/621735849)

**（4）ChatGPT AI 虚拟小镇**

[NPC 有生命了？ChatGPT AI 虚拟小镇，25 个 AI 居民的自由生活](https://www.bilibili.com/video/BV1vv4y1J7Li/)

虚拟小镇在线地址：https://reverie.herokuapp.com/arXiv_Demo/

论文地址：https://arxiv.org/pdf/2304.03442v1.pdf

类似的还有：[SkyAGI: Emerging human-behavior simulation capability in LLM](https://github.com/litanlitudan/skyagi)

这要是能做成虚拟现实的游戏，绝对爆火。

**（5）other**


[FastGPT](https://github.com/labring/FastGPT) ，一个基于 LLM 大语言模型的知识库问答系统

**（6）其他大语言模型**

参考：[机器学习/LLM](/ML/llm)



-------------------

参考资料：
- [InstructGPT 浅析](https://www.qin.news/instructgpt/)
- [Youtube - GPT社会化的过程](https://www.youtube.com/watch?v=e0aKI2GGZNg)
- [ChatGPT 原理剖析 (2/3) - 预训练](https://www.youtube.com/watch?v=1ah7Qsri_c8)
- [ChatGPT 原理剖析 (3/3) — ChatGPT 所带来的研究問題](https://www.youtube.com/watch?v=UsaZhQ9bY2k)


## 方法论

（1）当模型预测不准时，并且没有改进思路。可以查看预测不准的数据是什么样子，然后对数据筛选或寻找原因。




