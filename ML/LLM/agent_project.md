
主要记录一下 agent 相关的高星项目


## dify

应用层，用于快速构建 llm 应用

https://github.com/langgenius/dify



## n8n

workflow 层，用于快速构建一个任务流程;

集成较多插件，快速连接 google sheets, twitter 等；

自建的 agent 应用也推荐用 n8n 来获得多个应用交互能力，支持双向调用。

https://github.com/n8n-io/n8n


## openclaw


## automation

repo1. https://github.com/alibaba/page-agent

基于 dom 结构树，转化为文本数据然后通过（观察， 思考（拼接提示词）， 行动）操控浏览器。


IIFE 书签注入 （将立即执行函数表达式， 写入浏览器书签，点击书签时即可完成脚本注入）

常用于加载外部脚本(应用时压缩成一行):

```js
javascript:(function(){
  var s=document.createElement("script");
  s.src="https://example.com/tool.js";
  document.body.appendChild(s);
})();
```

-------------

repo2. https://github.com/bytedance/UI-TARS-desktop

基于图像理解