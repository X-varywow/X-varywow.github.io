

[微软 Win11/10 原版 ISO 光盘镜像下载大全](https://www.ithome.com/0/655/926.htm)

[Windows11专业版安装与激活](https://zhuanlan.zhihu.com/p/419103306)

另一款激活工具：https://github.com/TGSAN/CMWTAT_Digital_Edition

## 大概历程

更换SSD，开机检查，里面是空的，无法启动；

下载装机工具（大白菜、老毛桃都一个样，怀疑是一家公司），下载系统镜像，制作启动盘。

正常进PE，分区然后装系统。

BOOT启动项没有硬盘，但能检测到硬盘。重装好几次，还是这样。

UEFI 启动 和 lagacy 启动，传统 BIOS 启动改名为 lagacy 了。

最后是这个方案：分区表类 MBR -> GUID，才能正常 UEFI 启动

[硬盘分区表格式GUID和MBR知识普及](https://zhuanlan.zhihu.com/p/370552513)

## Windows PE

Preinstallation Environment，带有限服务的、最小 Win32 子系统，基于保护模式运行的 Windows 内核。



## Darwin

Darwin 是苹果公司开发的 UNIX 系统，是苹果所有系统的基础（包括 macOS）


------------

win11 是个大坑，不仅仅性能跑分比 win10 低。不设置 pin 直接登录界面死循环（至少几年了）。就像这个功能设计者买棺材需要亲人的入土证明，导致亲人入不了土还需要新开一个亲人的入土证明。

shift + 重启

[官方Windows 10镜像的靠谱下载点](https://zhuanlan.zhihu.com/p/81005418)


[U盘被分成2个盘怎么合并?](https://www.disktool.cn/content-center/usb-is-divided-into-two-disk-how-to-merge-369.html)

```bash
diskpart

lis dis

sel dis 3

clean
```