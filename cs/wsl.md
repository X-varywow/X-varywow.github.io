


windows 内部集成的 linux 子系统。

与 windows 深度互通，能通过 windows 接口访问和控制 windows 资源

现在 ai agent 自动化方案常用：

```bash
WSL:
    AI / Python / Shell / Agent
        ↓
Windows:
    AutoHotKey / PyAutoGUI
        ↓
Windows GUI 自动化
```


## 基本命令


```bash
wsl --install

# 列出可用的 Linux 分发版
wsl --list --online

# 列出已安装的 Linux 分发版
wsl --list --verbose

# 设置默认 WSL 版本
wsl --set-default-version <Version>
wsl --set-default-version 2

# 设置默认 Linux 分发版
wsl --set-default Ubuntu-22.04
```


```bash
sudo apt update && sudo apt upgrade
```






------------

参考资料：
- https://learn.microsoft.com/zh-cn/windows/wsl/