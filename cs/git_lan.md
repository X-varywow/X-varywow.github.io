

!>推荐阅读：[git 简易指南](https:#www.bootcss.com/p/git-guide/)

说明：环境为 `Windows 10`，代码中 `' '` 仅用于表征，实际输入时不用带。

git 是一个开源的分布式版本控制系统。



## 1.创建

```bash
git init          #将当前目录初始化成仓库

git init 'project_name'   #创建目录并初始化
```

## 2.检出
```bash
# 获取一个仓库的文件
git clone 'address' -b branch_name

# 获取远程最新分支，不会自动合并本地分支
git fetch origin branch_name

# 获取远程最新分支，并合并到本地分支
git pull origin branch_name
```

## 3.工作流
你的本地仓库由 git 维护的三棵“树”组成。

第一个是你的 `工作目录`，它持有实际文件；

第二个是 `缓存区（Index）`，它像个缓存区域，临时保存你的改动；

最后是 `HEAD`，指向你最近一次提交后的结果。

## 4.添加与提交
```bash
# 将其添加到缓存区

git add 'filename'

git add *

#将代码提交到 HEAD

git commit -m "代码提交信息"
```

## 5.推送改动

```bash
git remote add origin 'server'

git push origin 'branch'
```

## 6.分支 ⭐

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



## 7.更新与合并


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
```



</br>

### _git reset_

回退版本：

```bash
# 查看历史提交记录，记住 id 用于 reset
git log


git reset
```

[参考资料：reset](https://www.runoob.com/git/git-reset.html)


</br>

### _git rebase_

- 优点
  - 可以将多个小提交合并成一个大的提交，使提交历史更加清晰
  - 通过 rebase 拉取最新代码，避免合并冲突
  - 可以调整提交的顺序
- 缺点
  - 如果分支共享，rebase 改变提交历史，可能导致他人工作出现问题
  - 导致历史追溯问题


场景：两个分支 master 和 feature，

```bash
# 将 master 上的改动拉到本地 feature 分支
git checkout feature

git rebase master
```

推荐资料：https://www.freecodecamp.org/chinese/news/the-ultimate-guide-to-git-merge-and-git-rebase/

