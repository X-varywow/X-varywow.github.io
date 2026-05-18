

## windows 安装

windows 版本：（gpt: 不推荐， 适用性上 agent 是基于 linux 子系统设计的）

> 个人目前用着比 wsl 要舒服简单一些

```bash
# 走本地代理端口加速，这边是通过 _clash 启动了命令行；下载速度大提升
git clone https://github.com/NousResearch/hermes-agent.git


python -m venv .venv
.\.venv\Scripts\activate.bat
conda deactivate

pip install -e ".[all]"
```


wsl 版本：

```bash
# 安装 适用于 Linux 的 Windows 子系统（WSL）
wsl --install
wsl --install -d Ubuntu-22.04

# 在 wsl 中运行 或 cmd 中输入 wsl 进入
cd /mnt/c/Users/Administrator/Desktop/hermes-agent

# 复制或 clone 项目
# 正常 windows 桌面对应地址: /mnt/c/Users/Administrator/Desktop
cd ~
mkdir projects
cp -r /mnt/c/Users/Administrator/Desktop/hermes-agent ~/projects/
cd ~/projects/hermes-agent


wsl -l -v
# 进入 ubuntu
wsl -d Ubuntu-22.04

# windows 建立的不会兼容
rm -rf .venv

sudo apt update
sudo apt install -y python3-venv

sudo apt install -y python3.11 python3.11-venv python3.11-dev

mkdir -p ~/.pip
vim ~/.pip/pip.conf

# 粘贴如下内容
# [global]
# index-url = https://pypi.tuna.tsinghua.edu.cn/simple
# trusted-host = pypi.tuna.tsinghua.edu.cn
# timeout = 120

python3.11 -m venv .venv
source .venv/bin/activate

sudo apt install python3-pip
pip install -e ".[all]"

hermes
hermes update 
```


版本2，前面到 安装所需 pip 那里进行不下去了，网络实在下载太慢了

(**这里严重需要网络代理**)

```bash
# C:\Users\你的Windows用户名\.wslconfig
# 新增文件加入如下内容
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true


# 启动 _clash vpn 代理端口的命令行
# 重启 wsl
wsl --shutdown
wsl

cd ~/projects/
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```





## 启动

```bash
hermes          # 进入 CLI 交互
hermes --tui    # 进入 TUI 交互
hermes dashboard        # 打开浏览器仪表盘


hermes gateway      #Start messaging gateway
hermes doctor       #Check for issues
```

自用 windows 启动脚本：

```bash
cd C:\Users\Administrator\Desktop\hermes-agent
call .\.venv\Scripts\activate.bat
hermes gateway
pause
```

`call` 确保激活脚本在当前上下文中执行，而不是启动子进程

没有的话，父进程会被替换为子加成，activate.bat 执行完毕后整个脚本终止



## 常用命令

| 命令   | 作用                |
| ------ | ------------------- |
| /new   | strat a new session |
| /usage | token usage         |




## other

刷新了 soul, 但是微信机器人没有生效：session 里的 system prompt 是旧版的

`/new` 或 `/reset`

---------

如何清理记忆？问 hermes 她也说不清楚

```bash
hermes memory reset --yes 
# 在清理 /.hermes/sessions/ 下的文件
```
清memory, sessions, 微信记录都试了没有；

最后 保留 `.env` `SOUL.md` 清理掉 `.hermes`