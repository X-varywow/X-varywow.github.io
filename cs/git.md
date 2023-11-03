
!>推荐阅读：[git 简易指南](https:#www.bootcss.com/p/git-guide/)

说明：环境为 `Windows 10`，代码中 `' '` 仅用于表征，实际输入时不用带。

git 是一个开源的分布式版本控制系统。

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


## 基本操作 ⭐
##### 1. 创建新仓库

```bash
git init          #将当前目录初始化成仓库

git init 'project_name'   #创建目录并初始化
```

##### 2. 检出仓库
```bash
# 获取一个仓库的文件
git clone 'address' -b branch_name

# 获取远程最新分支，不会自动合并本地分支
git fetch origin branch_name

# 获取远程最新分支，并合并到本地分支
git pull origin branch_name
```

##### 3. 工作流
你的本地仓库由 git 维护的三棵“树”组成。

第一个是你的 `工作目录`，它持有实际文件；

第二个是 `缓存区（Index）`，它像个缓存区域，临时保存你的改动；

最后是 `HEAD`，指向你最近一次提交后的结果。

##### 4.添加与提交
```bash
# 将其添加到缓存区

git add 'filename'

git add *

#将代码提交到 HEAD

git commit -m "代码提交信息"
```

##### 5.推送改动

```bash
git remote add origin 'server'

git push origin 'branch'
```

##### 6.分支 ⭐

```bash
#创建一个叫做“feature_x”的分支，并切换过去：

git checkout -b feature_x

#切换回主分支：

git checkout master

#再把新建的分支删掉：

git branch -d feature_x
```


```bash
# 查看本地分支
git branch

# 查看远程分支
git branch -r


# 查看所有分支
git branch -a

# 删除本地分支
git branch -d localBranchName

# 删除远程分支
git push origin --delete remoteBranchName


# 创建一个分支，并切换回去
git checkout -b feature_x

# 推送至远程分支，没有就创建
git push orign branch_name
```

回退版本：

```bash
# 查看历史提交记录，记住 id 用于 reset
git log


git reset
```

[参考资料：reset](https://www.runoob.com/git/git-reset.html)


git rebase **修改分支提交的历史**，将一个分支的提交应用到另一个分支上，避免大量的合并提交。

场景：两个分支 master 和 feature，

```bash
# 将 master 上的改动拉到本地 feature 分支
git checkout feature

git rebase master
```

推荐资料：https://www.freecodecamp.org/chinese/news/the-ultimate-guide-to-git-merge-and-git-rebase/



##### 7.更新与合并
```bash
#更新本地仓库至最新改动

git pull orgin branch_name
```

将远端不同分支合并到本地当前分支：

```bash
git clone address -b test
cd repo_name
git branch
git fetch origin pro

# 将 pro merge 到当前本地分支
git merge origin/pro

:wq
```








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

6. 使用 tag

Tags are ref's that point to specific points in Git history。

可以使用 tag 保存重要的 repo （特定版本）

[参考](https://www.atlassian.com/git/tutorials/inspecting-a-repository/git-tag)

## 相关文件

</br>

_.gitignore_

进行 git 相关操作时，会跳过这些文件

```bash
*.png
*.jpg
*.zip
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



## Git 规范

参考：[大厂都在用的Git代码管理规范](https://mp.weixin.qq.com/s/6QxmajXJ9xuO_EpCtcaHsw)


| 分支    | 功能                      | 环境 | 可访问 |
| ------- | ------------------------- | ---- | ------ |
| master  | 主分支，稳定版本          | PRO  | 是     |
| develop | 开发分支，最新版本        | DEV  | 是     |
| feature | 开发分支，实现新特性      |      | 否     |
| test    | 测试分支，功能测试        | FAT  | 是     |
| release | 预上线分支，发布新版本    | UAT  | 是     |
| hotfix  | 紧急修复分支，修复线上bug |      | 否     |



## Github

#### 1. Github 查找仓库

1. 按照项目名/仓库名搜索（大小写不敏感）
`in:name xxx`

2. 按照README搜索（大小写不敏感）
`in:readme xxx`

3. 按照description搜索（大小写不敏感）
`in:description xxx`

4. stars数大于xxx
`stars:>xxx`

5. forks数大于xxx
`forks:>xxx`

6. 编程语言为xxx
`language:xxx`

7. 最新更新时间晚于YYYY-MM-DD
`pushed:>YYYY-MM-DD`

8. 找百科大全
`awesome xxx`

9. 找例子
`xxx sample`

10. 找空项目架子
`xxx starter`/`xxx boilerplate`

11. 找教程
`xxx tutorial`

#### 2. 关于LICENSE

<img src="http://www.ruanyifeng.com/blogimg/asset/201105/free_software_licenses.png" style="zoom:50%">


#### 3. Github Action

>GitHub Actions is a convenient CI/CD service provided by GitHub.


参考资料：
- [GithubAction---Workflow概念和基本操作](https://zhuanlan.zhihu.com/p/377731593)
- [GitHub Actions 入门教程](https://www.ruanyifeng.com/blog/2019/09/getting-started-with-github-actions.html)
- https://www.actionsbyexample.com/
- https://github.com/firstcontributions/first-contributions





## _一次完整流程_

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

例如我的， `-f` 是强制的意思，不然有时会报错；

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
