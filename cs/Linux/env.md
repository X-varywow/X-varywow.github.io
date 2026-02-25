

export 设置环境变量（对所有进程可见，设置的环境变量 **仅在当前会话中有效**）

```bash
export VARIABLE_NAME=VALUE
export APP_ENV=prod

# 添加到 PATH 环境变量中
export PATH=/usr/local/bin:$PATH

export PYTHONPATH=$PYTHONPATH:/Users/filename

# 打印环境变量
echo $PATH
```

```bash
# 显示所有环境变量
env

# 设置环境变量
env VAR = 1
```

------------

如果设置所有进程都可访问的环境变量，可以将变量添加到系统的全局配置文件中。常见的全局配置文件包括：

1. `/etc/profile`：适用于所有用户的登录 shell。
2. `/etc/bash.bashrc`：适用于所有用户的非登录 shell（仅对 Bash 有效）。
3. `/etc/environment`：适用于所有用户和所有进程。

例如，可以在 `/etc/environment` 文件中添加以下行：

```sh
VARIABLE_NAME=value

source /etc/environment
```


------------

windows 中设置：
```bash
set APP_ENV=prod
```
