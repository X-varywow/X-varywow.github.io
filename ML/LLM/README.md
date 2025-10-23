

llm

[llm介绍](https://mp.weixin.qq.com/s/kcGfK7ANUB2tbERjXAvbLw)


Copilot 核心差异是 Chunk策略、Embedding模型、以及召回策略调优，基座大模型相对而言护城河不深。

--------

202502, deepseek 5% 的资源，达到 gpt4o 的效果；实测更好

[deepseek-r1 介绍](https://mp.weixin.qq.com/s/HMvuzbEa_sysH-ItF0zVXg)；

强化的过程中涉及到对思考过程的奖励，

--------

|          | 回答质量 | 风险宽松 |
| -------- | -------- | -------- |
| deepseek | 很好     | 一般     |
| gpt      | 好       | 严格     |
| kimi     | 一般     |          |

感觉 gpt 4o 相较 ds, 带了更多私域的知识，通用问题上没有很大差异，私域上 ds 纯在泛泛而谈，而 4o 可以引出或了解更多私域问题。


--------


webui:
- https://github.com/ChatGPTNextWeb/NextChat
- https://github.com/open-webui/open-webui
- https://github.com/Bin-Huang/chatbox

https://docs.openwebui.com/

```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install open-webui

```

https://github.com/deepseek-ai/awesome-deepseek-integration/blob/main/README_cn.md