
主要记录一下 agent 相关的高星项目

https://github.com/topics/ai


## dify

应用层，用于快速构建 llm 应用

https://github.com/langgenius/dify



## n8n

workflow 层，用于快速构建一个任务流程;

集成较多插件，快速连接 google sheets, twitter 等；

自建的 agent 应用也推荐用 n8n 来获得多个应用交互能力，支持双向调用。

https://github.com/n8n-io/n8n


## openclaw

https://github.com/openclaw/openclaw



## hermes

The agent that grows with you

https://github.com/NousResearch/hermes-agent

- 社区评价（gpt）：适合长期保存上下文，累积知识库
- 能从成功任务自动生成 skill

--------


特别结构：SOUL.md（人格），MEMORY.md（长期知识），USER.md（你的画像），

适合个人知识管理，记录读书、总结思考等；




## page-agent

automation repo1. https://github.com/alibaba/page-agent

基于 dom 结构树，转化为文本数据然后通过（观察， 思考（拼接提示词）， 行动）操控浏览器。

------------

核心设计：
1. **Schema 约束**：`AgentOutput` 工具的 Zod schema 将反思字段与 action 放在同一结构中
2. **System Prompt 引导**：`<reasoning_rules>` 部分明确要求模型评估上步效果、检测循环、规划下步
3. **历史上下文**：每步的 `evaluation`、`memory`、`next_goal` 都被记录并在后续步骤中传递给 LLM



------------

IIFE 书签注入 （将立即执行函数表达式写入浏览器书签，点击书签时即可完成脚本注入）

常用于加载外部脚本(应用时压缩成一行):

```js
javascript:(function(){
  var s=document.createElement("script");
  s.src="https://example.com/tool.js";
  document.body.appendChild(s);
})();
```


## UI-TARS

automation repo2. https://github.com/bytedance/UI-TARS-desktop

基于图像理解