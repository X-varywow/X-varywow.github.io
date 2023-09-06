

[Nginx Proxy 配置](https://www.jianshu.com/p/8532461837b1)

## 基础

```bash
nginx -s reload
```


```bash
cat >/etc/nginx/conf.d/proxy.conf <<EOF
upstream aigc-ui {
    server localhost:7860       weight=5 max_fails=2 fail_timeout=8s;

}

server {
    location / {
        proxy_pass http://aigc-ui;
    }
}

EOF
```

> 这段代码用来配置 Nginx 代理服务器，将请求转发给 “aigc-ui” 的上游服务器
> 通过 proxy_pass 实现将所有请求转发的处理









## 工作链路


sagemaker 本地起服务

nginx 做内部代理

alb 做外部负载均衡的代理

[AWS ALB](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)


-----------------

init-nginx.sh

更新 repl

更新 nginx 配置

更新 proxy.conf， 添加 映射 server 域名





