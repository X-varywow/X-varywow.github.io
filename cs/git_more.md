

## 安装&配置

[git官网](https:#git-scm.com/)

```bash
git --version


# 查看帮助信息
git
```

```bash
git config --list

git config --global user.name 'name'

git config --global user.email 'email@address'
```

[git提示“warning: LF will be replaced by CRLF”的解决办法](https://blog.csdn.net/u012757419/article/details/105614028/)




## 其它操作

1. 差异化比较，红色表示删除的，绿色表示添加的

```bash
git diff 'source_branch' 'target_branch'
```

2. 内建的图形化git

```bash
gitk
```

3. 彩色的 git 输出

```bash
git config color.ui true
```

4. 显示历史记录时，只显示一行注释信息

```bash
git config format.pretty oneline
```

5. 交互地添加文件至缓存区

```bash
git add -i
```

## 相关文件

</br>

_.gitignore_

进行 git 相关操作时，会跳过这些文件

```bash
*.png
*.jpg
*.zip

# 会跳过 folder 下所有 read.txt 外的文件
folder/*
!folder/read.txt
```

</br>

_.gitattributes_

用于定义特定文件或路径的 Git 行为。它可以帮助 Git 在处理这些文件时应用特定的设置或规则。

- 语言或文件类型识别
- 指定文件换行符格式，Unix 的 LF，windows 的 CRLF
- 文件合并策略
- 文件属性设置
- 文件属性过滤

demo1:

```bash
*.unitypackage filter=lfs diff=lfs merge=lfs -text
```

demo2:

```bash
* text=auto eol=lf
```

- \* 表示匹配所有文件
- text=auto 自动检测文件类型为文本文件，则进行处理
- eol=lf 将文件的行尾标志转换为 LF 换行符


> windows 使用 CR回车符 LR换行符 作为换行标志，而 unix/linux 使用 LF 作为换行标志

------------


[LFS](https://git-lfs.com/)（large File Storage）, An open source Git extension for versioning large files



## 相关规范

参考：[大厂都在用的Git代码管理规范](https://mp.weixin.qq.com/s/6QxmajXJ9xuO_EpCtcaHsw)

</br>

_分支规范_


| 分支    | 功能                      | 环境 | 可访问 |
| ------- | ------------------------- | ---- | ------ |
| master  | 主分支，稳定版本          | PRO  | 是     |
| develop | 开发分支，最新版本        | DEV  | 是     |
| feature | 开发分支，实现新特性      |      | 否     |
| test    | 测试分支，功能测试        | FAT  | 是     |
| release | 预上线分支，发布新版本    | UAT  | 是     |
| hotfix  | 紧急修复分支，修复线上bug |      | 否     |

</br>

_Commit Message 规范_

`<type>(<scope>):<subject>`, 如 fix(a.py): divide by 0

常见 type 类型：
- feat: 新增功能
- fix: 修复 bug
- refactor: 不修复 bug 不添加特性的代码更改
- perf: 改进性能
- delete
- modify
- revert
- docs: 仅文档更改
- chore: 琐事
- style
- ci: 自动化流程配置更改
- test: 测试、测试用例等



## Github

#### 1. Github 查找仓库


大小写不敏感，

| 命令                        | 说明                  |
| --------------------------- | --------------------- |
| in:name xxx                 | 按照项目名/仓库名搜索 |
| in:readme xxx               |                       |
| in:description xxx          |                       |
| stars:>xxx                  |                       |
| forks:>xxx                  |                       |
| language:xxx                |                       |
| pushed:>YYYY-MM-DD          |                       |
| awesome xxx                 | 找百科大全            |
| xxx sample                  | 找例子                |
| xxx starter/xxx boilerplate | 空项目模版            |
| xxx tutorial                | 教程                  |


</br>

demo: 查找出现了 logger.addHandler 和 kinesis 的 python 代码

logger.addHandler kinesis language:Python



#### 2. 关于LICENSE

<img src="http://www.ruanyifeng.com/blogimg/asset/201105/free_software_licenses.png" style="zoom:50%">


#### 3. Github Action

>GitHub Actions is a convenient CI/CD service provided by GitHub.


参考资料：
- [GithubAction---Workflow概念和基本操作](https://zhuanlan.zhihu.com/p/377731593)
- [GitHub Actions 入门教程](https://www.ruanyifeng.com/blog/2019/09/getting-started-with-github-actions.html)
- https://www.actionsbyexample.com/
- https://github.com/firstcontributions/first-contributions





## _一次流程_

?> 这是将本 docsify 部署到 github pages 的流程。


（1）远程 github 配置

新建一个仓库，命名为：`<username>.github.io`


（2）将本地目录初始化成仓库，会生成一个 `.git` 的隐藏文件夹

```
git init
```

（3）将本地仓库连接到远程服务器，这样后面 fetch 和 push 时有了对象
```
git remote add origin <server>
```

（4）查看远程服务器信息
```
git remote -v
```

（5）更新本地仓库至最新改动
```
git pull
```

（6）将文件添加到缓存区
```
git add *
```

（7）提交修改
```
git commit -m "提交信息"
```

（8）push 到远程仓库
```
git push origin <分支名>
```

（9）配置 github pages

选择分支，并自定义域名（可选）

自定义域名配置之后会生成一个 `CNAME` 文件

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/202112122250067.jpg">

（10）自动化

>使用 vscode 内置的 Git 插件

>或者写一个 bat 脚本

`-f` 是强制的意思，否则有时会报错；

```bat
cd C:\docs
git add *
git commit -m "daily commit"
git push origin master -f
```

------------

参考资料：
- [阮一峰 - 最简单的 Git 服务器](https://www.ruanyifeng.com/blog/2022/10/git-server.html)
- https://www.atlassian.com/git/tutorials
- https://wiringbits.net/blog/github-repository-setup
