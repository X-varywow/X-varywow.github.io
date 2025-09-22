

!>推荐阅读：[git 简易指南](https://www.bootcss.com/p/git-guide/)

说明：环境为 `Windows 10`，代码中 `' '` 仅用于表征，实际输入时不用带。

git 是一个开源的分布式版本控制系统。



## 1.创建

```bash
git init          #将当前目录初始化成仓库

git init 'project_name'   #创建目录并初始化

git init --initial-branch=master
```

## 2.检出
```bash
# 获取一个仓库的文件
# 添加 local_name 可以直接命名本地检出的项目
git clone 'address' -b branch_name local_name

# 获取远程最新分支，不会自动合并本地分支
git fetch origin branch_name

# 获取远程最新分支，并合并到本地分支
git pull origin branch_name
```

demo: 使用远端分支创建出本地新分支
```bash
git fetch origin 

git checkout -b dev origin/dev

# 更改跟踪的远端分支
git branch --set-upstream-to=origin/test
```

demo: test 合并到 master 并不引起冲突
```bash
git fetch

git checkout test

git merge origin/master
```


## 3.工作流

你的本地仓库由 git 维护的三棵“树”组成。

第一个是你的 `工作目录`，它持有实际文件；

第二个是 `缓存区（Index）`，临时保存你的改动；

最后是 `HEAD`，指向你最近一次提交后的结果。



## 4.添加&提交&推送

```bash
# 将其添加到缓存区

git add 'filename'

git add *

#将代码提交到 HEAD

git commit -m "代码提交信息"
```


```bash
git remote add origin 'server'

git push origin 'branch'
```

--------------


默认情况下，对文件名称大小写是不敏感的，但是有些如 docsify 对特定命名要求大写；

方式1：重命名中转； 

方式2：修改配置：

```bash
# 查看 True 为忽视大小写
git config core.ignorecase

git config core.ignorecase false
```




## 5.分支 ⭐

```bash
# 创建一个叫做“feature_x”的分支，并切换过去：
git checkout -b feature_x

# 切换回主分支：
git checkout master

# 再把新建的分支删掉：
git branch -d feature_x

# -D 强制删除
git branch -D localBranchName

# 删除远程分支
git push origin --delete remoteBranchName
```


```bash
# 查看本地分支
git branch

# 查看远程分支
git branch -r

# 查看所有分支
git branch -a

# 推送至远程分支，没有就创建
git push origin branch_name
```



## 6.更新 pull


```bash
# 默认拉取本地对应的远端最新代码
git pull


#更新本地仓库至最新改动
git pull orgin branch_name
```

## 7.合并 merge

将远端不同分支合并到本地当前分支：

```bash
git clone address -b test
cd repo_name
git branch
git fetch origin pro

# 将 pro merge 到当前本地分支
git merge origin/pro

# 将本地 master merge 到当前分支（test）
git merge master

# --squash 合并 commit 信息
git merge --squash master
```

一般合并代码发生冲突，在编辑器中合并，接受传入;

------------

合并冲突时，所有冲突都选择当前分支（ours）的代码

```bash
git merge -X ours <branch_name>
```

------------

同时有 incoming 和 outgoing 变更，并且提示分支分叉(divergent branches)时，

合并方式 1. merge

```bash
git pull --no-rebase

# 或先设置默认行为再pull：
git config pull.rebase false
git pull
```


合并方式 2. rebase

```bash
git pull --rebase

# 或先设置默认行为再pull：
git config pull.rebase true
git pull
```

### rebase

- 优点
  - 历史干净、线性
  - 由于是把分支提交一个个摘下来再重放到新的基线上，可能导致冲突变多
- 缺点
  - 如果分支共享，rebase 改变提交历史，可能导致他人工作出现问题
  - 导致历史追溯问题


场景：两个分支 master 和 feature，

```bash
# 将 master 上的改动拉到本地 feature 分支
git checkout feature

git rebase master
```

### cherry-pick

复制 commit 到另一个分支上

```bash
# 查看历史 commit 信息
git log

git cherry-pick commit-id1 commit-id2
```


```bash
git revert commit-id
```


## 8.tag

Tags give the ability to mark specific points in history as being important

使用 tag 可以保存重要版本的代码

[参考](https://www.atlassian.com/git/tutorials/inspecting-a-repository/git-tag)

```bash
# 创建
git tag <tag_name> <commit_id>

# 推送远程
git push origin <tag_name>
git push origin --tags

# 删除
git tag -d <tag_name>
git push origin -d <tag_name>
```



## 9.other



### git log

```bash
git log

# 注意 -- 后有空格
git log -- file_name
git log -- dir_name

# 简化输出
git log --oneline

#查看每次提交详细差异
git log -p

# 限制输出数量
git log -5

# 
git log --graph

git log --help
# 回车：下一行，空格：翻页
```

可以查看老早的文件夹是谁建立的



### git blame

查看每行代码是谁写的

```bash
git blame <file-name>
```



### git reset

!> 执行特殊操作先备份文件

回退版本：

```bash
# 查看历史提交记录，记住 id 用于 reset
git log


git reset <commit_hash>
```

将分支回退到上一个提交：（上一个提交后的所有更改都会丢失）

```bash
git reset --hard HEAD~1
```

--hard 选项表示重置工作目录和索引，使其与新的 HEAD 状态完全一致

正常之后应该：

```bash
git push origin test -f
```

reset 之后 commits 就看不到中间的提交了


[参考资料：reset](https://www.runoob.com/git/git-reset.html)



### git stash ⭐️

存储临时代码。(当前需要切换分支且当前代码修改不像提交时)，**在 ide 中使用很便捷**

比如本地调试时，经常改一些代码，这时 stash 出来，后续 stash apply 就行。

```bash
# 保存当前未commit的代码
git stash save "备注内容"

# 按 q 退出
git stash list

# 应用最近一次的stash
git stash apply

# 应用对应的记录
git stash apply stash@{1}
```

发生冲突时，根据提示信息即可；

未跟踪文件 ---add--> 暂存区文件（向后 commit, 向前 restore --staged）



推荐资料：https://www.freecodecamp.org/chinese/news/the-ultimate-guide-to-git-merge-and-git-rebase/

