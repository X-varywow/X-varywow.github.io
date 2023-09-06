Uvicorn，用于 Python 的 ASGI Web 服务器。

Uvicorn 目前支持 HTTP/1.1 和 WebSockets。

## 1. ASGI

异步标准网关接口

参考：https://python.freelycode.com/contribution/detail/1827

## 2. 基础语法

参考：[官方文档](https://www.uvicorn.org/)

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

# --reload  enable auto-reload
```