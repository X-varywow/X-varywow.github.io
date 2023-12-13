
Python 内置了 requests 模块，该模块主要用来发 送 HTTP 请求，requests 模块比 urllib 模块更简洁。

```python
# 导入 requests 包
import requests

kw = {}

# 设置请求头
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}

# 发送请求
x = requests.get('https://www.runoob.com/', headers = headers, params = kw)

# 返回网页内容
print(x.text)

# 返回 http 的状态码
print(x.status_code)

# 响应状态的描述
print(x.reason)

# 返回编码
print(x.apparent_encoding)
```












--------

参考资料：
- [官方文档](https://docs.python-requests.org/en/latest/index.html) 
- [菜鸟教程](https://www.runoob.com/python3/python-requests.html)