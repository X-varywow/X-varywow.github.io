
## _logging_

logging -- 灵活的事件日志系统


（1）使用基本的 logging
```python
import logging

logging.basicConfig()
logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
```


（2）使用basicConfig
```python
logging.basicConfig(
    filename='application.log',
    level=logging.WARNING,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.error("Some serious error occurred.")
# [12:52:35] {<stdin>:1} ERROR - Some serious error occurred.
```

（3）高级
```python
FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
logger = logging.getLogger('tcpserver')
logger.warning('Protocol problem: %s', 'connection reset', extra=d)

#-->2006-02-08 22:20:02,165 192.168.0.1 fbloggs  Protocol problem: connection reset
```

（4）使用 logging 组件

- Logger
- Handler
- Filter
- Formatter

（5）异步日志操作

```python
import logging
import queue
log_queue = queue.Queue()

# 创建一个`QueueHandler`对象，并将其添加到根日志记录器中：
log_handler = logging.handlers.QueueHandler(log_queue)
root_logger = logging.getLogger()
root_logger.addHandler(log_handler)

# 自定义的异步日志处理器类
class AsyncLogHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
    def emit(self, record):
        log_queue.put(record)


async_log_handler = AsyncLogHandler()
log_listener = logging.handlers.QueueListener(log_queue, async_log_handler)
root_logger.addHandler(async_log_handler)

log_listener.start()
```

用起来报错：AttributeError: module 'logging' has no attribute 'handlers'.



-------------
参考资料：
- [官方文档](https://docs.python.org/zh-cn/3.9/library/logging.html)
- chatgpt



</br>

## _loguru_

相比传统的logging模块，loguru 更加适合现代Python项目的日志记录需求

- 使用简单
- 自动格式化，根据上下文自动确定消息格式
- 输出控制，级别、颜色等
- 自动轮转，避免单个日志文件过大的问题
- 异步日志，可以在后台线程中进行日志写入，不会阻塞主线程的执行


> 确实用起来要舒服很多

```python
from loguru import logger
import queue
import threading

log_queue = queue.Queue()

def log_worker():
    while True:
        record = log_queue.get()
        logger.opt(depth=6, exception=record.exc_info).log(record.levelno, record.getMessage())
        log_queue.task_done()

log_thread = threading.Thread(target=log_worker, daemon=True)
log_thread.start()

logger.add(log_queue.put, level="DEBUG")

# 使用示例
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
```

更多语法参考：https://github.com/Delgan/loguru


-------------

异步日志好处：
- **提高性能**，将日志记录与主程序的执行分离，不会阻塞主程序的执行，特别是高并发或高负载的场景
- **减少 IO 开销**，异步日志操作可以将多个日志请求合并成一个批量写入操作，减少磁盘 IO 次数（磁盘IO 是一个相对很慢的操作）
- **简化代码**，便于维护，将日志记录逻辑与主程序的逻辑分离