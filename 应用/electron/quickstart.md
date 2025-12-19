## 快速入门

[electron 快速入门例子](https://www.electronjs.org/zh/docs/latest/tutorial/quick-start)

安装好依赖后：

（1）进行项目初始化，并配置一些参数
```
npm init
```

（2）安装 electron 依赖
```
npm install --save-dev electron
```

（3）建立三个文件
- main.js 
  - electron 主进程
- preload.js
  - 通过预加载脚本从渲染器访问Node.js
- index.html
  - 应用展示出来的窗口

（4）运行 demo

package.json 中，脚本字段添加如下内容

```json
"start": "electron ."
```

`npm start` 即可完成运行


（5）打包分发

要使用管理员权限运行cmd

```
npm install --save-dev @electron-forge/cli
npx electron-forge import
npm run make
```

## 代码分析

entry point: main.js

（简易版）
```js
const {app, BrowserWindow} = require('electron')

const createWindow = () => {
    const win = new BrowserWindow({
        width: 720,
        height: 900
    })

    win.loadFile('index.html')
}

app.whenReady().then(() => {
    createWindow()
})
```


（官网摘录，）
```js
const {app, BrowserWindow} = require('electron')
const path = require('path')

// 创建窗口
function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })

  win.loadFile('index.html')
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

// darwin 是 macos 的代号
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
```