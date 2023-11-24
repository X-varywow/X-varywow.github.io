
## _javascript_


### 1. JS简介

- Web 的编程语言，控制着网页的行为。
- 可插入 HTML 页面

1. 在 浏览器 中使用 开发者工具，console 窗口可运行 Javascript 代码
2. 在 浏览器 中使用 开发者工具，sources 窗口可建立 Javascript 脚本

```javascript
console.log("hello, world")
```

- 使用 `;` 分隔语句
- 大小写敏感
- `//` 单行注释，`/**/` 多行注释


### 2. 数据类型&变量

#### 2.1 变量声明

- var 用于声明变量，无值的变量为 `undefined`
- let 声明的变量只能被定义一次
- const 声明的只读


#### 2.2 数据类型

- 基本类型
  - String
  - Number
  - Boolean...
- 引用类型
  - Object
  - Array
    - 存放的数据类型可以不同
  - Function...
- 检查方式
  - typeof xxx


#### 2.3 字符串

字符串可以用单引号或双引号，内部str中引号应与外部不同或使用`\`；

字符串可以使用 + 拼接

字符串不能通过索引赋值

str.length

### 3. 函数

- 输出
  - window.alert() ，弹出警告框
  - document.write()，写入 html 文档
  - innerHTML，写入到 html 元素
  - console.log()，写入到浏览器的控制台
- 函数

```javascript
function myfun(var1, var2){
    ...
}
```


</br>



## _ES6_

一个简单的回调函数（将函数作为参数传递）：

```js
const message = function() {
  console.log("This message is shown after 3 seconds");
  console.log(JSON.stringify(res))
}

setTimeout(message, 3000);
```

变成匿名函数：

```js
setTimeout(function(){
  console.log("This message is shown after 3 seconds");
}, 3000)
```

用箭头函数写回调函数：

```js
setTimeout( () => {
  console.log("This message is shown after 3 seconds");
}, 3000)
```


箭头函数，使用示例1：

```js
// 使用 () 包裹对象，以区分代码块

var f = (id,name) => ({id: id, name: name});
```




箭头函数，使用示例2：

```js
const holistic = new Holistic({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }
});
```
locateFile 是一个函数名，在这里是可以省略的。

由于箭头函数被直接传递给 Holistic 构造函数，locateFile 前不需要使用 const 或 var 来声明。


</br>

## _浏览器插件_

需求：检测到特定界面，修改界面样式 

使用插件 stylus, 直接里面自定义样式，可对特定页面生效


[浏览器插件制作教程1](https://xieyufei.com/2021/11/09/Chrome-Plugin.html)


</br>

## _other_

.js 文件是纯粹的 JavaScript 文件，它只包含 JavaScript 代码，没有任何模块化的语法或依赖管理。

.umd.js 文件是通用模块定义（Universal Module Definition）的 JavaScript 文件。UMD 是一种用于实现跨平台模块化的技术，它兼容多种模块加载器（如 CommonJS、AMD、Node.js 等），可以在不同的环境中使用。UMD 文件通常包含了对不同模块加载器的适配代码，以确保在不同环境下都能正确加载和执行模块。

总结来说，.js 文件仅包含纯 JavaScript 代码，而 .umd.js 文件是经过兼容多种模块加载器的处理，可以在不同环境中使用的 JavaScript 文件。



--------------

参考资料：
- https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/
- https://www.runoob.com/js/js-tutorial.html
- https://javascript.info/
- [runoob es6-function](https://www.runoob.com/w3cnote/es6-function.html)
- [javascript-callback-functions](https://www.freecodecamp.org/chinese/news/javascript-callback-functions/)
- chatgpt