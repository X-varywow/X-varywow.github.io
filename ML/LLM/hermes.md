

## windows 安装


```bash
# 走本地代理端口加速，这边是通过 _clash 启动了命令行；下载速度大提升
git clone https://github.com/NousResearch/hermes-agent.git


# 安装 适用于 Linux 的 Windows 子系统（WSL）
wsl --install

# 在 wsl 中运行 或 cmd 中输入 wsl 进入
cd /mnt/c/Users/Administrator/Desktop/hermes-agent


# 有点麻烦，要装完全版 wsl; 换 Windows cmd 了
python -m venv .venv
.\.venv\Scripts\activate.bat
conda deactivate

pip install -e ".[all]"
```


wsl 版本：
```bash
wsl --install -d Ubuntu-22.04

wsl -l -v
# 进入 ubuntu
wsl -d Ubuntu-22.04

# 设置默认 wsl
wsl --set-default Ubuntu-22.04

# windows 建立的不会兼容
rm -rf .venv

sudo apt update
sudo apt install -y python3-venv


cd ~
mkdir projects
cp -r /mnt/c/Users/Administrator/Desktop/hermes-agent ~/projects/
cd ~/projects/hermes-agent


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
pip install -e ".[all]"

hermes
hermes update 

```



## 启动

```bash
hermes          # 进入 CLI 交互
hermes --tui    # 进入 TUI 交互
hermes dashboard        # 打开浏览器仪表盘


hermes gateway      #Start messaging gateway
hermes doctor       #Check for issues
```

## TODO

- [ ] Windows 切到 wsl





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