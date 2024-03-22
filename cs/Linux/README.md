欢迎来到 Linux 首页

参考资料：
- [菜鸟教程Linux](https://www.runoob.com/linux/linux-tutorial.html)
- [Linux 工具快速教程](https://linuxtools-rst.readthedocs.io/)
- 跟阿铭学Linux
  - 4：Linux 文件和目录管理
  - 5：系统用户与用户组管理
  - 6：Linux 磁盘管理



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


windows 中设置：
```bash
set APP_ENV=prod
```