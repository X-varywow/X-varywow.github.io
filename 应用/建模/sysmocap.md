
参考论文：[《视频驱动虚拟角色动作的自动生成系统的设计与实现》](https://mp.weixin.qq.com/s/TJd6LBbtHHhKAksFwXrERg)

参考博客：https://blog.xianfei.eu.org/p/bishe/

仓库地址：https://github.com/xianfei/SysMocap/tree/main

[GitHub搜 motion capture](https://github.com/search?q=%20motion%20capture&type=repositories)

作者 2022 年毕业的，blog 更新的少啊，摄影确实也是艺术。



## preface

`使用技术栈：`
- 前端
  - electron
  - vue
  - https://github.com/xianfei/fxdrawer
  - threejs
  - https://github.com/pixiv/three-vrm
- 图像处理
  - mediapipe （也可用 tf lite 代替，[参考](https://blog.tensorflow.org/2021/08/3d-pose-detection-with-mediapipe-blazepose-ghum-tfjs.html)）

`依赖版本：`
- electron v25.4
- nodejs v18.15


`优点：`
- 界面美观合理，electron 跨平台
- 支持 OBS, 通过端口 socket 转发实现
- 支持更多的 3D 格式：VRM, GLB, FBX （需要包含骨骼信息）
- 从0构建整套流程，UI, 挺厉害的


`缺点：`
- 占用很高，mocap 运行在 14FPS （部分是 electron 本身导致的，运行浏览器内核导致程序性能不好）
- 抖动问题较明显
- 相机视角有问题，无法建立稳定的对应关系
- 对于 fbx 文件，并不友好，
- 综上缺点，输入为视频文件时mocap只有 8FPS，导致效果极差，略过了很多帧。（没kalidoface web 版效果好，网页版也是，手部很麻烦，图层深度跟没有一样）



## 文件结构

需要预装 nodejs, 然后如下运行

```bash
git clone https://github.com/xianfei/SysMocap.git
cd SysMocap
npm i
npm start
```

入口文件 main.js，在 package.json 中 "main": "main.js"，"start": "electron ."指定，负责 `createWindow()`

内置了个 pdfviewer ，还挺好看


- main/framework 
  - 基本几个界面的前端：setting,render, titlebar
- render
  - 同 mocap render
- mocap
- mocaprender
  - 包含 animateVRM
- models
- modelview
- utils
- webserv




只使用了 mediapipe 的 holistic, kalidokit




## vs kalidoface VRM

仓库地址：https://github.com/yeemachine/kalidokit

在线体验：https://glitch.com/edit/#!/kalidokit

与之前的 kalidokit 类似，为啥不用纯 web? 性能应该会更好，


## 流程






## 一些代码

### web 3D 支持

支持 VRM（专注人型，统一的骨骼标准等， 基于 gltf）gltf fbx 

通过 threejs 作为 3D 渲染引擎


### electron ipcmain & 创建窗口

```js
//事件监听
ipcMain.on("openDocument", function (event, arg) {
    createPdfViewerWindow(__dirname + "/pdfs/document.pdf");
});
```


```js
function createPdfViewerWindow(args) {
    // Create the browser window.
    var viewer = new BrowserWindow({
        width: 1180,
        height: 750,
        autoHideMenuBar: true,
        titleBarStyle: "hidden",
        trafficLightPosition: { x: 10, y: 8 },
        titleBarOverlay: {
            color: "#0000",
            symbolColor: nativeTheme.themeSource=='dark'?'#eee':'#111',
        },
        // vibrancy: "dark",
        webPreferences: {
            nodeIntegration: true,
            webviewTag: true,
            contextIsolation: false,
            enableRemoteModule: true,
            additionalArguments: ["pdfPath", JSON.stringify(args)],
        },
    });

    viewer.webContents.on('will-navigate', (e, url) => {
        e.preventDefault()
        shell.openExternal(url)
    })
    // 处理 window.open 跳转
    viewer.webContents.setWindowOpenHandler((data) => {
        shell.openExternal(data.url)
        return {
            action: 'deny'
        }
    })

    // and load the index.html of the app.
    viewer.loadFile("pdfviewer/viewer.html");
    electronRemoteMain.enable(viewer.webContents);

    // Open the DevTools.
    // viewer.webContents.openDevTools()

    // Emitted when the window is closed.
    viewer.on("closed", function () {
        viewer = null;
    });
}
```

## other

.github/workflows/main.yml git打包发布自动化工作流

.vscode/launch.json 配置vscode执行信息

.gitignore





-----------

参考资料：
- sysmocap - document