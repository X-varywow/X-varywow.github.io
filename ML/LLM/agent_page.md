

https://github.com/alibaba/page-agent


## other

IIFE 书签注入 （将立即执行函数表达式， 写入浏览器书签，点击书签时即可完成脚本注入）

常用于加载外部脚本(应用时压缩成一行):

```js
javascript:(function(){
  var s=document.createElement("script");
  s.src="https://example.com/tool.js";
  document.body.appendChild(s);
})();
```