## _注册表_

注册表是windows操作系统、硬件设备以及客户应用程序得以正常运行和保存设置的核心“数据库”，也可以说是一个非常巨大的树状分层结构的数据库系统。

注册表记录了用户安装在计算机上的软件和每个程序的相互关联信息，它包括了计算机的硬件配置，包括自动配置的即插即用的设备和已有的各种设备说明、状态属性以及各种状 态信息和数据。利用一个功能强大的注册表数据库来统一集中地管理系统硬件设施、软件配置等信息，从而方便了管理，增强了系统的稳定性

参考资料：[windows注册表](https://blog.csdn.net/wz_cow/article/details/88835569)

- 右键菜单对应的注册表位置：
  - 计算机\HKEY_CLASSES_ROOT\\*\shell
  - 计算机\HKEY_CURRENT_USER\Software\Microsoft\Internet Explorer\MenuExt\
- 新建选项卡 
  - 计算机\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Discardable\PostSetup\ShellNew


---------------

**注册表编辑器**

- HKEY_CLASSES_ROOT
- HKEY_CURRENT_USER
- HKEY_LOCAL_MACHINE
- HKEY_USERS
- HKEY_CURRENT_CONFIG



---------------

参考资料：
- [纯粹的Windows右键菜单管理程序](https://github.com/BluePointLilac/ContextMenuManager)⭐
- [Win10管理右键新建菜单](https://blog.csdn.net/weixin_44811846/article/details/103288139)
- [自定义右键菜单](https://shliang.blog.csdn.net/article/details/89286118)


## _windows命令_

| 命令             | 描述                  |
| ---------------- | --------------------- |
| `cd`             | 查看当前路径（鸡肋）  |
| `cls`            | 清空窗口              |
| `help`           |                       |
| `dir`            |                       |
| `TREE`           | 显示目录结构          |
| `COPY`           |                       |
| `MOVE`           |                       |
| `DEL`            |                       |
| `MD`             | 创建文件夹            |
| `RD`             | 删除文件夹            |
| `echo %COMSPEC%` | 查看使用的是什么shell |

## _批处理文件_
- 一次性运行一批CMD命令，可以写在文本文档中
- 以`bat`为后缀名，双击可执行


例子：一个完成 hexo 自动部署的 bat

```cmd
cd /blog
hexo cl & hexo g & hexo d
```

## _开机自动启动服务_

1. 将所需自动执行的 bat 文件放入 win10 启动文件夹，
`C:\Users\你的用户名\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup`
2. 这时候，开机便会执行这个文件，但还是会出现黑窗口
3. 将第一步的文件放入另一个地方，新建 vbs 文件（执行bat），代码如下：
```vbs
set ws=WScript.CreateObject("WScript.Shell")
ws.Run "C:\\Users\\User\\Documents\hexo-server.bat /start",0
```
