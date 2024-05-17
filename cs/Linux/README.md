欢迎来到 Linux 首页

---------------

`man` 显示命令的帮助

`history` 显示历史指令



| 按键     | 描述         |
| -------- | ------------ |
| `TAB`    | 自动补全命令 |
| `CTRL+C` | 终止当前命令 |
| `CTRL+D` | 退出终端     |
| `↑`      | 上一条指令   |


```bash
# 查看内核版本

cat /proc/version
```

------------

export 设置环境变量（对所有进程可见，设置的环境变量 **仅在当前会话中有效**）

```bash
export VARIABLE_NAME=VALUE
export APP_ENV=prod

# 添加到 PATH 环境变量中
export PATH=/usr/local/bin:$PATH

export PYTHONPATH=$PYTHONPATH:/Users/filename


echo $PATH

# 导出变量
export VARIABLE_NAME
```

如果设置所有进程都可访问的环境变量，可以将变量添加到系统的全局配置文件中。常见的全局配置文件包括：

1. `/etc/profile`：适用于所有用户的登录 shell。
2. `/etc/bash.bashrc`：适用于所有用户的非登录 shell（仅对 Bash 有效）。
3. `/etc/environment`：适用于所有用户和所有进程。

例如，可以在 `/etc/environment` 文件中添加以下行：

```sh
VARIABLE_NAME=value

source /etc/environment
```


windows 中设置：
```bash
set APP_ENV=prod
```


------------

参考资料：
- chatgpt
- [菜鸟教程Linux](https://www.runoob.com/linux/linux-tutorial.html)
- [Linux 工具快速教程](https://linuxtools-rst.readthedocs.io/)
- 跟阿铭学Linux
  - 4：Linux 文件和目录管理
  - 5：系统用户与用户组管理
  - 6：Linux 磁盘管理