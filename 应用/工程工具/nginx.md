

>Nginx (engine x) 是一款轻量级的 Web 服务器 、反向代理服务器及电子邮件（IMAP/POP3）代理服务器。

反向代理、负载均衡


## 基础

```bash
# 更改相关配置后 reload
nginx -s reload
```


```bash
cat >/etc/nginx/conf.d/proxy.conf <<EOF
upstream site_name {
    server localhost:7860       weight=5 max_fails=2 fail_timeout=8s;

}

server {
    location / {
        proxy_pass http://site_name;
    }
}

EOF
```

> 这段代码用来配置 Nginx 代理服务器，将请求转发给 “site_name” 的上游服务器
> 通过 proxy_pass 实现将所有请求转发的处理



```bash
# 测试 nginx 配置文件语法是否正确
nginx -t

systemctl start nginx

# 检查 nginx 状态
sudo systemctl status nginx

```





## 链路

一种可行的链路：sagemaker 本地起服务，nginx 做内部代理，alb 做外部负载均衡的代理

[AWS ALB](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)


-----------------


init-nginx.sh （在 sagemaker 上使用）

（1）准备环境

```bash
yum -y install epel-release

yum -y install nginx
```

（2）更新 nginx 配置

/etc/nginx/nginx.conf

（3）更新 proxy.conf， 添加 映射 server 域名

/etc/nginx/conf.d/proxy.conf


## 遇坑

/etc/nginx/conf.d/proxy.conf 某行代码如下：

```bash
proxy_set_header   Upgrade $http_upgrade;
```

引起报错：nginx: [emerg] invalid number of arguments in "proxy_set_header" directive in /etc/nginx/conf.d/proxy.conf:9

原因：没有转义变量；但是以前是这样跑起来的，，，

解决方法：

```bash
proxy_set_header   Upgrade \$http_upgrade;
```


-------------

参考资料：
- [Nginx 极简教程](https://github.com/dunwu/nginx-tutorial)
- [Nginx Proxy 配置](https://www.jianshu.com/p/8532461837b1)