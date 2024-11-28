

## _curl_

curl 是常用的命令行工具，用来请求 Web 服务器。它的名字就是客户端（client）的 URL 工具的意思。

它的功能非常强大，命令行参数多达几十种。如果熟练的话，可替代 Postman 这一类的图形界面工具。

```bash
# 普通 GET 请求
curl https://www.example.com

# -X 用于指定 HTTP 请求方法, 默认的是 GET 方法; 同 --request

# -d 发送 POST 请求的数据体
curl -d 'login=emma＆password=123' -X POST https://google.com/login 


# -o 将服务器的回应保存成文件，等同于wget命令。
curl -o example.html https://www.example.com
```

```bash
# -A 指定 User-Agent

# -b 向服务器发送 Cookie

# -e 设置 HTTP 的标头Referer
curl -e 'https://google.com?q=example' https://www.example.com

# -H 添加 HTTP 请求的标头
curl -H 'Accept-Language: en-US' https://google.com

# -u 设置服务器认证的用户名和密码。

# -L 指定 curl 跟随重定向，用于跳转下载且不好找下载地址的地方; 同 --location
curl -L https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr -o /content/so-vits-svc/hubert/checkpoint_best_legacy_500.pt
```


其它：

```bash
# -o- 表示输出到标准输出，即 curl 不会保存文件而是将下载的数据输出到终端
# | bash 将上层管道传过来的数据直接 bash 执行
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
```

---------

参考资料：
- [curl 的用法指南](https://www.ruanyifeng.com/blog/2019/09/curl-reference.html)









</br>

## _wget_

wget 是一个从网络上自动下载文件的自由工具。它支持HTTP，HTTPS和FTP协议，可以使用HTTP代理。

常见用法：

```bash
# 下载 SD 底模 + LORA 微调
wget https://civitai.com/api/download/models/94640 -O ./models/Stable-diffusion/majicmixRealistic_v6.safetensors
wget https://civitai.com/api/download/models/102236 -O ./models/Lora/cartoon_portrait_v1.safetensors
```

| 参数    | 说明                     |
| ------- | ------------------------ |
| -q      | 安静模式，不显示输出     |
| -O file | 写到file文件中           |
| -o file | 追加写到file文件中       |
| -P      | 指定下载文件时保存的目录 |



</br>

## _netstat_


`netstat` 是一个网络工具，用于显示网络连接、路由表、接口统计等网络相关信息。它通常用于诊断网络问题和监控网络状态。以下是一些常见的 `netstat` 使用场景和选项：

- **查看所有活动的网络连接**：
```
netstat -a
```

- **查看所有监听的端口**：
```
netstat -l
```

- **查看所有已建立的连接**：
```
netstat -c
```

- **查看路由表**：
```
netstat -r
```

- **查看网络接口的使用情况**：
```
netstat -i
```

- **显示详细的连接信息**：
```
netstat -n
```

- **显示持续时间**：
```
netstat -o
```

- **显示程序的PID和名称**：
```
netstat -p
```

- **显示TCP连接状态**：
```
netstat -t
```

- **显示UDP连接**：
```
netstat -u
```

- **显示所有选项的组合**：
```
netstat -x
```

请注意，`netstat` 命令在不同的操作系统中可能略有不同，并且某些选项可能需要管理员权限才能使用。此外，`netstat` 命令在一些现代操作系统中已经被 `ss` 命令所取代，后者提供了更多的功能和更好的性能。


## _demo_

grep 用于在文本中搜索指定的模式

wc 字数统计工具，wc -l 只统计行数

```bash
netstat -na | grep ".8888"

netstat -na | grep ".8888" | grep ESTABLISHED | wc -l
```

-------------

```bash
#显示套接字网络信息
ss -tunlp
```