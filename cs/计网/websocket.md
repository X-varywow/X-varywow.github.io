WebSocket 是一种网络传输协议，可在单个TCP连接上进行全双工通信，位于应用层。


</br>

特殊功能：
- 允许服务端主动向客户端推送数据。
- 在 WebSocket API 中，浏览器和服务器只需要完成一次握手，两者之间就可以建立持久性的连接，并进行双向数据传输。


`WS`（WebSocket）和 `WSS`（WebSocket Secure）是两种网络通信协议。

- `WS` 是WebSocket的简写，它是一种在单个长连接上进行全双工通信的协议。它允许服务器和客户端之间发送文本和二进制消息，主要用于浏览器和服务器之间的交互。`WS`的URL格式类似于http，例如：`ws://example.com/`。

- `WSS` 是WebSocket Secure的简写，是`WS`的扩展，它在WebSocket基础上添加了SSL/TLS加密层。这使得客户端和服务器之间的通信被加密，对抗eavesdropping（窃听）和man-in-the-middle攻击。`WSS`的URL格式类似于https，例如：`wss://example.com/`。

简单来说，`WSS`比`WS`更安全，通常用于需要加密通信的场景。


</br>

## _安全性_

Websocket与HTTP和HTTPS **使用相同的TCP端口**，可以绕过大多数防火墙的限制。 

默认情况下，Websocket协议使用 **80端口** ；运行在TLS之上时，默认使用 **443端口**。

在生产环境上使用 websocket 本身是安全的：
- 使用 WSS
- 验证连接请求，防止未授权的访问
- 记录访问日志，便于调查攻击
- 跨站点脚本保护 (XSS)
- 安全实践




</br>

## _other_


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


--------------

记一次报错：

部署 streamlit 服务时：

`WebSocket connection to 'wss://***.com/_store/stream' failed: main.js:2`

解决方案：开启 websocket
