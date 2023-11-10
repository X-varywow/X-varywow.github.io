
用于在 python 中实现 <u>可靠性和重试逻辑</u> 的模块, 常用于处理网络请求、数据库操作或其他可能出现失败的操作。


- 灵活的重试策略
- 支持指数退避算法
- 异常处理
- 超时处理
- 支持异步操作




```python
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def make_network_request():
    # 进行网络请求的代码
    response = requests.get('https://example.com')
    response.raise_for_status()
    return response.json()
```

```python
@retry(stop=(stop_after_delay(10) | stop_after_attempt(5)))
def stop_after_10_s_or_5_retries():
    print("Stopping after 10 seconds or 5 retries")
    raise Exception
```








----------

参考资料：
- https://tenacity.readthedocs.io/en/latest/
- chatgpt