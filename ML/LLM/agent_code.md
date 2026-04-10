


## 基本概念

https://github.com/tukuaiai/vibe-coding-cn


## AGENTS.md

ai 编码指南，在对话中，自动追加到当前的上下文信息中，帮助 ai 熟悉本项目的结构和基本规则。

核心内容：项目结构、规范， 对具体业务还应该包含：业务流程、业务（代码）术语

> 坚持原则：一切问 ai, 然后手动提示。让 ai 编写，让 ai 问询关键信息，让 ai 评价， 让 ai 改进 AGENTS.md


--------

一次业务项目中，ai 对 AGENTS.md 的评价：
- 节省了 AI 探索代码库理解上下文的时间，加权平均提效 35% （减少具体问题所需对话轮次）
- 准确率平均绝对提升 20%
- AGENTS.md 的核心价值是把 AI 从"每次从零探索"变成"带着正确心智模型开始工作", ROI 较高



--------

**指导性提示词**：
- When you have a good idea to improve this file, ask the user and make changes as needed.
- Finding the simplest solution possible, and only increasing complexity when needed.
- Every change you make should not only implement the desired functionality but also improve the quality of the codebase.
- All code and comments must be in English.
- Do not try to hide errors or risks. They are valuable feedback for developers and users. Make them visible and actionable.


--------

范例推荐：
- https://github.com/alibaba/page-agent/blob/main/AGENTS.md
- https://github.com/n8n-io/n8n/blob/master/AGENTS.md



## skills

[cursor skills docs](https://cursor.com/cn/docs/skills)

[腾讯技术工程-agent_skills](https://mp.weixin.qq.com/s/ho1l5v5mrNr_f6JXARMlFQ)


可以直接通过对话使其生成 skill

文件结构:

```bash
my-skill/
├── SKILL.md          # 核心：指令 + 元数据
├── scripts/          # 可选：可执行代码
├── references/       # 可选：参考文档
└── assets/           # 可选：模板、资源
```




## commands


```bash
.cursor/
    |-skills/
    |-commands/
        |-command1.md
```

本质是写一段提示词在 md 中，然后通过 `/command1` 在 聊天窗口运行。


常见用处：
- 自定义 `/note` 来将本次对话整理到合适的笔记中，方便后续查看。
- 自定义 `/session:logger` `/session:restorer` 来将本次对话上下文存储和读取，用于跨对话上下文传递。
- 自定义 `/requirement:new` `/requirement:save` `/requirement:list` 进行需求管理。
- 自定义 `/solo` （根据代码或提示）快速构建业务中可独立运行的代码，如查库（让 ai 拼接出上下文）用于快速 debug，文档中追加一些示例和错误示例。




--------

参考资料：
- [tencent-Agentic Engineering](https://mp.weixin.qq.com/s/ri_lxDGayM-e5A0oAW59Fw)