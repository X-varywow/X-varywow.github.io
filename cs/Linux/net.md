

## _curl_

curl 是常用的命令行工具，用来请求 Web 服务器。它的名字就是客户端（client）的 URL 工具的意思。

它的功能非常强大，命令行参数多达几十种。如果熟练的话，完全可以取代 Postman 这一类的图形界面工具。

```bash
# 普通 GET 请求
curl https://www.example.com


# -d 发送 POST 请求的数据体
curl -d'login=emma＆password=123'-X POST https://google.com/login


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

# -L 指定 curl 跟随重定向，用于跳转下载且不好找下载地址的地方，救大了
curl -L https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr -o /content/so-vits-svc/hubert/checkpoint_best_legacy_500.pt
```

</br>

## _wget_

wget 是一个从网络上自动下载文件的自由工具。它支持HTTP，HTTPS和FTP协议，可以使用HTTP代理。

常见用法：

```bash
# 下载 SD 底模 + LORA 微调
wget https://civitai.com/api/download/models/94640 -O ./models/Stable-diffusion/majicmixRealistic_v6.safetensors
wget https://civitai.com/api/download/models/102236 -O ./models/Lora/cartoon_portrait_v1.safetensors
```

| 参数    | 说明                 |
| ------- | -------------------- |
| -q      | 安静模式，不显示输出 |
| -O file | 写到file文件中       |
| -o file | 追加写到file文件中   |


---------

参考资料：
- [curl 的用法指南](https://www.ruanyifeng.com/blog/2019/09/curl-reference.html)