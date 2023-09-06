
> express 是一后端框架（基于 nodejs），主要用于处理 web后端的请求和响应。

问了下 chatgpt, 除 express 最近几年的服务框架:

<p class="warn">
- Koa：Koa是由Express团队开发的下一代Node.js框架。它采用了更现代的异步编程风格，使用了ES6的Generator函数和async/await语法糖。Koa具有更轻量级的设计和更灵活的中间件系统。</br></br>
- NestJS：NestJS是一个基于TypeScript的渐进式Node.js框架，它结合了Angular的开发风格和Express的灵活性。NestJS提供了强大的依赖注入、模块化和面向对象的编程模式，使得构建可扩展的应用程序变得更加容易。</br></br>
- Fastify：Fastify是一个高性能的Web框架，专注于提供快速和低开销的API服务。它采用了异步编程风格，具有出色的性能和低内存消耗。Fastify支持插件系统，可以轻松地扩展功能。
</p>


## hello world


第一步：初始化 nodejs 项目，并安装 express

```bash
mkdir myapp
cd myapp


npm init

sudo npm install express
```

第二步：新建文件

```js
// index.js

const express = require('express')
const app = express()
const port = 3000

app.get('/', (req, res) => {
  res.send('Hello World!')
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
```

修改 package.json

```json
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "serve": "node index.js"
  },
```

第三步：运行服务

```bash
npm run serve
```

如果报错，切换默认端口


> 好吧，网页服务的入口可以是 js，说以前 html 怎么找不到 js 的引入



## getting started

```bash
sudo npm install -g express-generator
```




## 模板引擎

> 由于 express 创建项目时需要指定模板引擎，这里简要说明


`pug` 是一个模版引擎，与 hexo 默认的 `ejs` 类似。使用简洁的缩进和动态代码来创建模版，并能够减少HTML代码的重复性。


express 的模板语法为什么不使用 vue?

?> Express框架提供的view engine（视图引擎）是用于 **服务器端**，因此没有内置 vue </br></br>
Vue 主要用于构建用户界面，可以与任何后端框架配合，</br>
在实际开发中，可以将Express与Vue结合使用，通过Express提供API和数据，然后使用Vue来构建动态的用户界面。


[hexo-theme-butterfly](https://github.com/jerryc127/hexo-theme-butterfly) 中就使用到了 pug，语法非常简洁

[hexo-theme-matery](https://github.com/blinkfox/hexo-theme-matery) 中使用的模板语法是 ejs，看低来要复杂得多。



----------

参考资料：
- [官方文档](https://expressjs.com/en/starter/installing.html)
- chatgpt