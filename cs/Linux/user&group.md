
```shell
# 创建用户组
groupadd [-g GID] groupname

# 删除用户组
groupdel groupname

# 用户
usderadd [-u UID] [-g GID] [-d HOME] [-M] [-s]

# -r 选项：一并删除家目录
userdel [-r] username
```

```shell
# 切换到 root 用户
su
# 退出
exit

# 执行一个高权限的命令
sudo

# 切换到 user2 并使用其环境变量
su - user2
```

[添加环境变量](https://blog.51cto.com/u_14782715/5082236)

```bash
​​$PATH="$PATH":YOUR_PATH​​
```

```bash
​​export PATH="$PATH:YOUR_PATH"​​
```
