

## windows

#### 系统工具

| 用途                | 工具                                                                         |
| ------------------- | ---------------------------------------------------------------------------- |
| 系统激活            | [CMWTAT_Digital_Edition](https://github.com/TGSAN/CMWTAT_Digital_Edition)    |
| Office 一键安装激活 | [LKY_OfficeTools](https://github.com/OdysseusYuan/LKY_OfficeTools)           |
| 卸载 / 注册表清理   | [Geek Uninstaller](https://geekuninstaller.com/)                             |
| 右键菜单管理        | [ContextMenuManager](https://github.com/BluePointLilac/ContextMenuManager) ⭐ |
| 经典右键菜单        | Windows 11 Classic Context Menu                                              |
| 自动登录 Win11      | [修改注册表方式](https://zhuanlan.zhihu.com/p/411167130)                     |

#### 日常软件

| 用途          | 工具                                     |
| ------------- | ---------------------------------------- |
| 视频播放      | [PotPlayer](https://potplayer.daum.net)  |
| EPUB 阅读     | JustRead                                 |
| 游戏帧率监控  | Afterburner / Xbox Game Bar              |
| 游戏画质优化  | NVIDIA GeForce Experience                |
| Switch 模拟器 | [YUZU](https://github.com/yuzu-emu/yuzu) |

#### 快捷键

| 快捷键          | 说明         |
| --------------- | ------------ |
| `Win` + `Space` | 切换美式键盘 |

> 文件管理器地址栏可直接输入 `cmd` 打开终端。

#### 美化

| 用途           | 工具                                                           |
| -------------- | -------------------------------------------------------------- |
| 任务栏透明     | TranslucentTB                                                  |
| 桌面整理       | [Stardock Fences](https://www.stardock.com/products/fences/)   |
| 自定义鼠标     | Yolo Mouse                                                     |
| 任务栏增强     | ExplorerPatcher                                                |
| 系统字体更换   | [noMeiryoUI](https://github.com/Tatsu-syo/noMeiryoUI)          |
| 右键菜单自定义 | [教程](https://shliang.blog.csdn.net/article/details/89286118) |

**字体推荐：**
- 系统字体：微软雅黑 9pt
- 浏览器字体：苹方 / SF Pro（苹果字体风格）
- 衬线字体：[思源宋体](https://source.typekit.com/source-han-serif/cn/)
- [修改系统字体教程](https://zhuanlan.zhihu.com/p/601288823)




## mac

#### 快捷键

| 快捷键                | 说明            |
| --------------------- | --------------- |
| `Cmd` + `Shift` + `3` | 截全屏          |
| `Cmd` + `Shift` + `4` | 区域截图        |
| `Cmd` + `Shift` + `5` | 录屏 / 截图选项 |
| `Cmd` + `Space`       | 聚焦搜索        |
| `Cmd` + `M`           | 最小化窗口      |
| `Cmd` + `W`           | 关闭窗口        |

#### 推荐软件

| 用途     | 软件                                       |
| -------- | ------------------------------------------ |
| 输入法   | 搜狗输入法                                 |
| 窗口管理 | [Magnet](https://magnet.crowdcafe.com)     |
| 办公     | [LibreOffice](https://www.libreoffice.org) |

#### 终端配置

**核心组合**：iTerm2 + zsh + Oh My Zsh + Nerd Fonts + Dracula 配色

- [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh)
- [Nerd Fonts](https://github.com/ryanoasis/nerd-fonts)
- [Dracula Theme](https://github.com/dracula/dracula-theme)

```zsh
# 编辑配置
vim ~/.zshrc
# 主题设置
ZSH_THEME="agnoster"
```

#### Homebrew

```bash
# 安装
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 在 ~/.zshrc 中添加，然后 source ~/.zshrc
export PATH=/opt/homebrew/bin:$PATH
```


iTerm2 快捷键

| 快捷键      | 说明       |
| ----------- | ---------- |
| `Cmd` + `T` | 新建标签页 |








## common

#### 效率工具

| 用途     | 工具                                                                              |
| -------- | --------------------------------------------------------------------------------- |
| 文档翻译 | 谷歌翻译                                                                          |
| 网页翻译 | [Immersive Translate](https://github.com/immersive-translate/immersive-translate) |
| 思维导图 | Xmind                                                                             |
| 阅读     | 微信读书 / NeatReader                                                             |
| 论文下载 | Sci-Hub                                                                           |

#### 硬件

| 用途     | 说明                                                     |
| -------- | -------------------------------------------------------- |
| 多键鼠标 | 自定义按键映射，如：复制、粘贴、左右删除                 |
| 虚拟机   | Windows 用 VMWare；Mac 用 VirtualBox / Parallels Desktop |


## vpn

目前使用的是 **clash**

(1) 教程，流程上通用

https://help.ghelper.net/shou-ji-dai-li/clash-for-windows

印象中，中间漏了一步：右键 profile -> run script，才会出现 proxies


(2) 代理端口

clash 界面端口编号那里可以直接打开代理过的端口，git 等操作会加速很多


(3) 如何让 vscode, cursor 使用代理？

对于单个项目，新建 settings.json

```json
{
  "http.proxy": "http://127.0.0.1:7890",
  "http.proxySupport": "override",
  "http.systemCertificates": true
}
```