
## _空间管理_

```bash
# 查看磁盘空间占用情况 (disk free)
df -h

# - T 显示文件系统类型
# - i 不用硬盘容量，而用 inode 的数量显示


# 查看文件或目录的占用情况（disk usage）
du

# 显示当前目录的占用磁盘容量
du -ah --max-depth=1

# 磁盘分区
fdisk

# 查看内存，并以 MB 显示
free -m

# 或者使用其他工具：duf, 可以可视化符号、彩色地显示磁盘情况
```

- total 总可用物理内存
- free  可用物理内存
- available 可被 应用 使用的物理内存， = free + buffer + cache

linux 为了提升读写性能，会缓存一部分内存资源缓存磁盘数据。

参考：[linux free 命令下free/available区别](https://blog.csdn.net/gpcsy/article/details/84951675)
</br>

## _设备信息_

```bash
# cpu 信息
cat /proc/cpuinfo

# 内存信息
cat /proc/meminfo

# 内存详细信息
sudo dmidecode -t memory

# 内核版本
cat /proc/version
```

</br>

## _开销信息_

```bash
# 每 1 s 显示一次显存
watch -n 1 nvidia-smi

# 提供由Linux内核管理的所有当前运行任务的动态实时统计汇总。它监视 Linux 系统上进程、CPU 和内存的完整利用率
top

# shift + m 根据内存降序
# shift + p 根据cpu降序

free -g

history
```

[nvidia-smi 更多的介绍](https://blog.csdn.net/C_chuxin/article/details/82993350)