
参考项目：
- 阿里云盘
- COSBrowser
- [picgo](https://github.com/Molunerfinn/PicGo)
- [electron-vue](https://github.com/SimulatedGREG/electron-vue)

涉及技术：
- 反编译
- electron
- 腾讯云 API

>萌新的 electron 之旅，electron、 js 都处于 quick-start 水平  `2022.01.13` <br> 想做一个 COSBrowser，picgo 结合一起的东西，picgo 删除、查看图片都不方便。开始吧~


### 1. 反编译

参考文章：
- [Electron程序逆向（asar归档解包）](https://www.cnblogs.com/cc11001100/p/14290584.html)

以阿里云盘的反编译为例：

#### 1.1 如何识别 electron 项目

- 风格上：浓浓的网页风
- 目录结构上：
  - locales
  - resources
    - app.asar
  - swiftshader
  - license.electron.txt
  - licenses.chromium.html
  - v8_context_snapshot.bin

#### 1.2 反编译

安装 asar 库

```
npm install --engine-strict asar
```

这里，我安装之后直接跑到当前用户的 node_modules 文件夹里了。按教程的方法跑不出来，这是我的：

将阿里云盘 resources 下的所有文件移动到 `C:\Users\User\node_modules\asar\bin` 中
![](https://img-1301102143.cos.ap-beijing.myqcloud.com/202201131824599.jpg)

再在 bin 路径下执行如下指令：
```
node asar.js e app.asar aliyunpan
```

最后会生成一个 aliyunpan 的文件夹，就是我们需要的代码了


### 2. 框架

