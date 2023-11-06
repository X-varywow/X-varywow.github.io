
Node.js 就是运行在服务端的 JavaScript。

Node.js 是一个事件驱动 I/O 服务端 JavaScript 环境，基于 Google 的 V8 引擎，V8 引擎执行 Javascript 的速度非常快，性能非常好。


```bash
node -v
```

```bash
# 初始化项目
npm init 
```

修改 package.json

```json
{
  "name": "myapp",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "serve": "node index.js"
  },
  "author": "",
  "license": "ISC",
  "dependencies": {
    "express": "^4.18.2"
  }
}
```


```bash
# 安装依赖
npm install

# 启动服务
npm run serve
```



------------

参考资料：
- [菜鸟教程](https://www.runoob.com/nodejs/nodejs-tutorial.html)