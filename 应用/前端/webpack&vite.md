

## _webpack_

webpack 是一个前端项目构建工具，基于nodejs开发。具有如下功能：

- **模块打包**：将项目中的各个模块（包括JavaScript、CSS、图片等）打包成一个或多个静态资源文件，以便在浏览器中加载和使用。
- **代码分割**：将代码拆分成多个异步加载的块（chunk），从而实现按需加载，减小初始加载的文件大小，提高页面加载速度。
- 资源优化
- 模块化支持


_示例：_

目前有个简单项目：包含 module.js、main.js

module.js:
```js
export const message = "Hello, World!";
```

main.js:
```js
import {message} from './module.js'

console.log(message)
```

第一步：配置 webpack, webpack.config.js 中定义入口文件和输出文件
```js
const path = require('path');

module.exports = {
    entry: './main.js',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.js'
    }
}
```

第二步：安装 webpack 和加载器，配置加载器

```bash
npm install webpack webpack-cli --save-dev
npm install babel-loader @babel/core @babel/preset-env --save-dev
```

在 webpack.config.js 的 exports 中添加：

```js
module: {
    rules: [
        {
            test: /\.js$/,
            exclude: /node_modules/,
            use: {
                loader: 'babel-loader',
                options: {
                    presets: ['@babel/preset-env']
                }
            }
        }
    ]
}
```

第三部：打包

```bash
npx webpack
```

会根据配置文件打包成 bundle.js



---------

参考资料：
- [Webpack基本使用（详解）](https://juejin.cn/post/6844903892031897608)
- https://webpack.js.org/
- https://webpack.js.org/guides/getting-started/
- chatgpt



</br>

## _vite_

Vite，[下一代前端开发利器——Vite（原理源码解析）](https://zhuanlan.zhihu.com/p/475176203)

[Vite - 新型前端构建工具](https://cn.vitejs.dev/guide/)，替代 webpack