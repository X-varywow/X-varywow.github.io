> cubism sdk 对 unity 支持得较多，高扩展性。Native 支持编码设计画面

[各Cubism SDK的比较](https://docs.live2d.com/zh-CHS/cubism-sdk-manual/cubismsdk-compare/)


## web sdk

[下载 SDK](https://www.live2d.com/zh-CHS/download/cubism-sdk/download-web/)

参考：[构建Web范例](https://docs.live2d.com/zh-CHS/cubism-sdk-tutorials/sample-build-web/)


打开 SDK 下的 Samples/TypeScript/Demo，运行如下指令即可

```bash
sudo npm install

# 将源代码打包为可部署的文件
# dist 中输出 bundle.js
sudo npm run build

npm run serve
```

有些别扭，需要在界面中打开正确文件夹才看到界面。

> 熟悉了 nodejs 配置项目（package.json）,管理配件（node_modules），构建项目并本地起服务的方式。局域网可访问？

## unity sdk

https://docs.live2d.com/zh-CHS/cubism-sdk-tutorials/getting-started/

加载 Cubism SDK（.unitypackage文件）

载入模型，会在 model_name/runtime/ 下自动生成 预制体
