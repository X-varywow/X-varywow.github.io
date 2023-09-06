
参考示例：[Electron + Vue3 开发跨平台桌面应用【从项目搭建到打包完整过程】](https://juejin.cn/post/6983843979133468708)


### 0. 工具链说明

在几年前，使用 electron 和 vue 来开发应用使用的是 [electron-vue](https://github.com/SimulatedGREG/electron-vue) ，现在跑路了...

现在 vue 引入 electron 一般是 [vue-cli-plugin-electron-builder](https://github.com/nklayman/vue-cli-plugin-electron-builder)，[参考文档](https://nklayman.github.io/vue-cli-plugin-electron-builder/)

Vite，[下一代前端开发利器——Vite（原理源码解析）](https://zhuanlan.zhihu.com/p/475176203)

原本构建工具推荐使用的是 Vite，但现在 electron 引入 vue 社区大多用的是 vue-cli。

所以本文使用 vue-cli


### 1. 创建项目

```bash
npm i @vue/cli -g

vue create project_name

cd project_name
vue add electron-builder
```

报的错好多

1. [更改nodejs 文件夹权限](https://blog.csdn.net/fernwehseven/article/details/122337440)

2. [0308010C:digital envelope routines::unsupported](https://blog.csdn.net/zjjxxh/article/details/127173968)，OpenSSL3.0对允许算法和密钥大小增加了严格的限制


安装好之后
```bash
npm run electron:serve
```