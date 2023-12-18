
## _preface_

Python 内置了 requests 模块，该模块主要用来发 送 HTTP 请求，requests 模块比 urllib 模块更简洁。

```python
# 导入 requests 包
import requests

kw = {}

# 设置请求头
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}

# 发送请求
response = requests.get('https://www.runoob.com/', headers = headers, params = kw)

# 返回网页内容
print(response.text)

# 返回 http 的状态码
print(response.status_code)

# 响应状态的描述
print(response.reason)

# 返回编码
response.apparent_encoding
response.encoding
```

```python
data = {
    "name": "mike"
}

# TCP 数据包重传窗口，connect() 默认为 3
# requests 默认会一直等待，timeout 指定超时时间
res = requests.post(url, data = data, timeout = 5)

res.text

res.json()
```

## _长连接_

> HTTP 1.1 是长连接通信方式；除此， HTTP 1.0 + Connection: keep-alive 也可以指定长连接

keep-alive is 100% automatic within a session! 

Any requests that you make within a session will <u> automatically reuse the appropriate connection. </u>

默认值 10 个 长连接

class requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0, pool_block=False)


```python
import requests
s = requests.Session()
a = requests.adapters.HTTPAdapter(max_retries=3)
s.mount('http://', a)
```

## _session_

HTTP 协议本身是无状态的；为了让请求保持状态，有了 session 和 cookie


```python
import requests

s = requests.Session()
# 第一步：发送一个请求，用于设置请求中的cookies
# tips: http://httpbin.org能够用于测试http请求和响应
s.get('http://httpbin.org/cookies/set/sessioncookie/123456789')
# 第二步：再发送一个请求，用于查看当前请求中的cookies
r = s.get("http://httpbin.org/cookies")
print(r.text)
```




--------

参考资料：
- [官方文档](https://docs.python-requests.org/en/latest/index.html) 
- [菜鸟教程](https://www.runoob.com/python3/python-requests.html)
- https://www.cnblogs.com/zhuosanxun/p/12679121.html
- [构建高效的python requests长连接池](https://xiaorui.cc/archives/4437)，感觉有点错误，使用 session 会提高性能