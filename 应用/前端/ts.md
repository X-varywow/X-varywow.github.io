
## preface

https://www.typescriptlang.org/

TypeScript is a strongly typed programming language that builds on JavaScript, giving you better tooling at any scale.

TypeScript 可以看成 JavaScript 的超集。



!> TypeScript 是一种静态类型的编程语言，需要编译成 js 才能在浏览器或 nodejs 等环境运行，不进行编译是无法直接运行的。</br></br>
常见的前端工具（Webpack, Create React App, Vue CLI）可以自动执行 TypeScript 编译这个过程。



## hello world



要新建一个 TypeScript 项目，你可以按照以下步骤进行操作：

1. 确保你已经安装了 Node.js 和 npm（Node.js 包管理器）。

2. 打开命令行工具，进入你想要创建项目的目录。

3. 运行以下命令来创建一个新的空项目文件夹：

```bash
mkdir myproject
cd myproject
```

4. 初始化新的 npm 项目，使用以下命令：

```bash
npm init
```

按照提示输入一些基本信息，例如项目名称、版本号、描述等。你也可以直接按回车键接受默认值。

5. 安装 TypeScript 作为开发依赖项，使用以下命令：

```bash
npm install typescript --save-dev
```

或者：

```bash
npm install -g typescript

tsc -v
```



6. 创建一个 `tsconfig.json` 文件来配置 TypeScript 项目。运行以下命令来生成一个默认的 `tsconfig.json` 文件：

```bash
npx tsc --init
```

这个命令会在当前目录下生成一个默认的 `tsconfig.json` 文件。

7. 在项目文件夹中创建你的 TypeScript 文件，例如 `app.ts`。你可以使用任何文本编辑器打开这个文件，并输入你的 TypeScript 代码。

```js
const hello:string = "Hello World!"
console.log(hello)
```


8. 编译 TypeScript 文件为 JavaScript 文件。运行以下命令来编译 `app.ts`：

```bash
npx tsc app.ts
```

这个命令会将 `app.ts` 编译为 `app.js`。

9. 在你的项目中使用编译后的 JavaScript 文件。你可以使用任何支持 JavaScript 的环境来运行 `app.js`，例如 Node.js 或者浏览器。

可以使用 node 命令直接执行 app.js

```bash
node app.js
```





----------------

参考资料：
- https://wangdoc.com/typescript/
- https://www.runoob.com/typescript/ts-tutorial.html
- chatgpt


示例项目：
- [tabby: A terminal for a more modern age](https://github.com/Eugeny/tabby) https://tabby.sh/ (ts 能写出这种东西，第一次在 mac 上见到 pkg 带的安装器)