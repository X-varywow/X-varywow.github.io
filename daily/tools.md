
## mac 

| 快捷键                           | 说明     |
| -------------------------------- | -------- |
| `command` + `shift` + `3`        | 截全屏   |
| `command` + `shift` + `4`        | 截屏     |
| `command` + `shift` + `5`        |          |
| `command` + `space`              | 聚焦搜索 |
| `command` + `m`, `command` + `w` |          |

搜狗输入法

窗口管理软件 magnet

显示器排列

iterm2 + zsh + [ohmyzsh](https://github.com/ohmyzsh/ohmyzsh) + [nerd fonts](https://github.com/ryanoasis/nerd-fonts) + [dracula-theme](https://github.com/dracula/dracula-theme) 配色

```zsh
vim ~/.zshrc
```

```zsh
# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"
export PATH=/opt/homebrew/bin:$PATH
# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded, in which case,
# to know which specific one was loaded, run: echo $RANDOM_THEME
# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="agnoster"
[ -f /opt/homebrew/etc/profile.d/autojump.sh ] && . /opt/homebrew/etc/profile.d/autojump.sh

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/hua/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/hua/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/hua/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/hua/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


# Set list of themes to pick from when loading at random
# Setting this variable when ZSH_THEME=random will cause zsh to load
# a theme from this variable instead of looking in $ZSH/themes/
# If set to an empty array, this variable will have no effect.
# ZSH_THEME_RANDOM_CANDIDATES=( "robbyrussell" "agnoster" )

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
# HYPHEN_INSENSITIVE="true"

# Uncomment one of the following lines to change the auto-update behavior
# zstyle ':omz:update' mode disabled  # disable automatic updates
# zstyle ':omz:update' mode auto      # update automatically without asking
# zstyle ':omz:update' mode reminder  # just remind me to update when it's time

# Uncomment the following line to change how often to auto-update (in days).
# zstyle ':omz:update' frequency 13

# Uncomment the following line if pasting URLs and other text is messed up.
# DISABLE_MAGIC_FUNCTIONS="true"

# Uncomment the following line to disable colors in ls.
# DISABLE_LS_COLORS="true"
```

将 ~/.bash_profile 中的 conda 配置移到 ~/.zshrc

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/hua/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/hua/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/hua/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/hua/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
export PATH=/opt/homebrew/bin:$PATH
```

在 `~/.zshrc` 中添加 export, 然后 `source ~/.zshrc`

除了更改文件的方式，还可以使用如下命令

```bash
echo 'export PATH="/opt/homebrew/opt/ncurses/bin:$PATH"' >> ~/.zshrc
```




------------


brew 安装一些奇奇怪怪的东西：

- [trippy](https://trippy.cli.rs/), 一个命令行工具，可以代替 traceroute 查看互联网通信的路径，分析网络状况



```bash
export PATH=/usr/local/python-2.7.6/bin:$PATH 
```

autojump， 快捷地文件跳转

```bash
brew install autojump

vim  ~/.zshrc
# 添加如下：
# [ -f /opt/homebrew/etc/profile.d/autojump.sh ] && . /opt/homebrew/etc/profile.d/autojump.sh

source ~/.zshrc

# 查看数据库
autojump --stat
j --stat

# 使用 j 跳转到指定目录
j csm
# /data/www/xxx/cms
```








iterm2 快捷键：

| 快捷键      | 说明    |
| ----------- | ------- |
| command + T | new tab |




office 办公：libreoffice




>mac 确实不好用，系统限制多，软件兼容性没 windows, 卡卡的关闭还要好久。m1 弄个 live2d 打开模型直接CPU吃满；只是下限高，所以很多人说 mac 好，

--------------------



更多资料：
- [mac怎么快速回到桌面 隐藏所有窗口](https://www.cnblogs.com/guchunchao/p/9771548.html)


## windows

[系统激活](https://github.com/TGSAN/CMWTAT_Digital_Edition)

[一键自动化 下载、安装、激活 Office](https://github.com/OdysseusYuan/LKY_OfficeTools)，666

stardock fences 4

clash for windows

https://git.crepe.moe/taiga74164/Akebi-GC/-/tree/master

potplayer 播放器

文件管理器 path 里可直接 cmd

epub 阅读器：justread

[YUZU switch 模拟器](https://github.com/yuzu-emu/yuzu)

> Afterburner 或 xbox game bar 查看游戏帧率，开销等

> NVIDIA Gefore experience 进行游戏画面设置调整


`Win` + `space` 切换美式键盘

--------------------

### windows美化

TranslucentTB 任务栏透明

ExplorerPatcher

系统字体还是 微软雅黑   9pt， 浏览器的改一改，改成苹果字体

[修改系统字体](https://zhuanlan.zhihu.com/p/601288823)

[思源宋体](https://source.typekit.com/source-han-serif/cn/)

- [字体更换工具](https://github.com/Tatsu-syo/noMeiryoUI)
- [纯粹的Windows右键菜单管理程序](https://github.com/BluePointLilac/ContextMenuManager)⭐
- [自定义右键菜单](https://shliang.blog.csdn.net/article/details/89286118)
- [win 11 自动登录](https://zhuanlan.zhihu.com/p/411167130)，修改注册表方式，adminstrator 可使用 windows 的密码自动登录


Windows 11 Classic Context Menu 使用经典右键菜单


## common

谷歌翻译，用于文档翻译

飞书文档

xmind

微信读书，NeatReader

多按键的鼠标，定义多个按键功能，eg. 复制、粘贴、左右删除

虚拟机：Windows:VMWare, Mac: VirtualBox、Parallels Desktop 


## IDE 通用

[anaconda 镜像下载](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D)


| 快捷键           | 说明                     |
| ---------------- | ------------------------ |
| cmd + D          | 选定一个单词             |
| cmd + C, cmd + V |                          |
| cmd + X          | 删除行                   |
| cmd + Z          |                          |
| cmd + click      | 查看函数定义，被引用位置 |


vscode:
- word count
- indent-rainbow
- 设置：自动保存
- workbench>tree indent: 20
- audio: 0
- Editor: Format On Save
- LCPR: leetcode

> 使用 console.log(document.cookie) 可直接获取网页的 cookie

插件 EditorConfig for VS Code 统一代码样式

.vscode 下会有各项的 json 配置
- extensions.json
- launch.json
- settings.json
- tasks.json


settings.json

```json
{
    "window.zoomLevel": 1,
    "editor.tabSize": 4,
    "editor.formatOnSave": true
}
```

launch.json （run & debug 相关信息）

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "APP_ENV": "test"
            }
        }
    ]
}
```

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "启动 Electron",
      "type": "node",
      "request": "launch",
      "cwd": "${workspaceFolder}",
      "runtimeExecutable": "npm",
      "windows": {
        "runtimeExecutable": "npm.cmd"
      },
      "args": ["start"],
      "outputCapture": "std"
    }
  ]
}
```



tasks.json

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "启动 Electron 应用",
      "type": "shell",
      
      // 执行命令，会从 package.json 定义的 start 启动
      "command": "npm start",

      // 定义为构建任务，设置为默认任务
      "group": { 
        "kind": "build",
        "isDefault": true
      },

      // 展示方式；总是新启面板
      "presentation": {
        "reveal": "always",
        "panel": "new"
      },
      "problemMatcher": []
    }
  ]
}
```

可使用 ctrl + shift + b 运行 task

ctrl + shift + p 打开 vscode 魔法命令行， 如 `>Tasks: Run Task`


command + shift + p `>Open User Settings (JSON))`

```json
    "terminal.integrated.fontSize": 14,
    "terminal.integrated.fontFamily": "Fira Code",
    "terminal.integrated.lineHeight": 1.2
```




字体：[FiraCode](https://github.com/tonsky/FiraCode)


## 调试

`Step into`

执行下一行代码，进入函数调用。


`Step over`

执行下一行代码，但不进入任何函数调用。

如果当前行是一个函数调用，调试器会执行整个函数，但不进入该函数内部。


`Step out`

用于从当前函数退出，执行完当前函数的剩余部分，并停在函数的返回点上。