
一款原神抽卡记录分析工具

代码复现，来自：https://github.com/biuuu/genshin-wish-export

参考仓库：
- [原神签到小助手](https://github.com/y1ndan/genshinhelper2)
- [GenshinUID 3.1](https://github.com/KimigaiiWuyi/GenshinUID)


参考资料：
- [GitHub Actions 入门教程](https://www.ruanyifeng.com/blog/2019/09/getting-started-with-github-actions.html)
- [GithubAction---Workflow概念和基本操作](https://zhuanlan.zhihu.com/p/377731593)

需要学习：
- electron
- vite
- tailwind （一个功能类优先的 CSS 框架）
- 基础前端（html css js）


## 1. 原理

### 1.1 文件说明

- **.electron-vite** `vite工具`
  - build.js
  - dev-runner.js
  - rollup.main.config.js
  - update.js
  - vite.config.js
- .github
  - wordkflows
    - build-update.yml
    - release.yml
- build `放了一个ico，没啥`
- docs ~说明文档和实例图片~
- dist `构建时生成的文件`
- **src**
  - i18n `label 与 文本对应的json`
  - main
  - renderer
  - web
- package.json


```cmd
npm install -g yarn

yarn --version

yarn install
#报错1，换了源还是安装不了
#报错2，vpn 还是不行，更新nodejs 可以...
```


```bash
npm i @vue/cli -g

vue create tasky-vue

cd  tasky-vue

vue add electron-builder
```