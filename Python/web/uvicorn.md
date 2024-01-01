Uvicorn，用于 Python 的 ASGI Web 服务器。

Uvicorn 目前支持 HTTP/1.1 和 WebSockets。

## 1. ASGI

异步标准网关接口

ASGI（Asynchronous Server Gateway Interface）是一种Python的Web服务器接口规范。

它是在WSGI（Web Server Gateway Interface）的基础上发展而来，旨在解决Python Web应用程序在高并发场景下的性能问题。



## 2. 基础语法

```bash
pip install uvicorn
```

```python
# example.py
async def app(scope, receive, send):
    assert scope['type'] == 'http'


    # 使用协程与服务器通信
    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [
            [b'content-type', b'text/plain'],
        ],
    })
    await send({
        'type': 'http.response.body',
        'body': b'Hello, world!',
    })
```

```bash
# 运行服务器
uvicorn example:app

# --reload  开启热重载
```

访问 `http://127.0.0.1:8000/docs` (由 swagger ui 自动生成的交互式 api 文档)



----------

参考资料：
- https://python.freelycode.com/contribution/detail/1827
- [官方文档](https://www.uvicorn.org/)