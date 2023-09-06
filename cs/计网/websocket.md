WebSocket 是一种网络传输协议，可在单个TCP连接上进行全双工通信，位于应用层。


</br>

特殊功能：
- 允许服务端主动向客户端推送数据。
- 在 WebSocket API 中，浏览器和服务器只需要完成一次握手，两者之间就可以建立持久性的连接，并进行双向数据传输。


Websocket与HTTP和HTTPS使用相同的TCP端口，可以绕过大多数防火墙的限制。 默认情况下，Websocket协议使用80端口；运行在TLS之上时，默认使用443端口。


-----------


记一次报错：

使用 sagemaker 作为服务器，nginx 做内部代理，alb 做外部代理，使用域名访问服务时，

出现如下报错：

`WebSocket connection to 'wss://***.com/queue/join' failed: index.js:474`




初步分析：大概率是端口访问限制的问题


解决方案：https://github.com/gradio-app/gradio/issues/3716



```python
async def ws_login_check(websocket: WebSocket) -> Optional[str]:
    token = websocket.cookies.get("access-token") or websocket.cookies.get(
        "access-token-unsecure" 
    )
```

只知道这样解决了，好像更改 nginx 也行
