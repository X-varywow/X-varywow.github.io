
traceback ，用于处理和打印异常的跟踪信息。

当程序发生异常时，traceback 模块可以帮助开发者定位异常的发生位置，以及提供异常的堆栈跟踪信息。



```python
import tracebook

class demo:
    def _keep_alive(self, interval):
        while True:
            try:
                self.check()
            except Exception:
                error = traceback.format_exc()
                logger.error(f"Failed by: {error}")
            finally:
                time.sleep(interval)
```

参考： [官方文档](https://docs.python.org/zh-cn/3/library/traceback.html)