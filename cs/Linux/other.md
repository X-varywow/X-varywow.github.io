

sshfs 磁盘挂载

tcpdump 查看某个网站的 TCP 通信



</br>

## _文件打包_

```bash
# 仅打包，不压缩
tar -cvf archive_name.tar /path/to/directory

# -z 表示使用 gzip 压缩
tar -czvf archive_name.tar.gz /path/to/directory
```

</br>

## _定时任务_

`crontab` 用于设置周期性执行指令， cron 是一个守护进程
