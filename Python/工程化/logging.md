
## _logging_

logging -- 灵活的事件日志系统

日志，是对软件执行时所发生事件的一种追踪方式。


### （1）使用基本的 logging

```python
import logging

logging.basicConfig()
logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
```


### （2）使用basicConfig

```python
# 将日志写入到文件中

logging.basicConfig(
    filename='application.log',
    level=logging.WARNING,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.error("Some serious error occurred.")
# [12:52:35] {<stdin>:1} ERROR - Some serious error occurred.
```

### （3）高级

```python
FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
logger = logging.getLogger('tcpserver')
logger.warning('Protocol problem: %s', 'connection reset', extra=d)

#-->2006-02-08 22:20:02,165 192.168.0.1 fbloggs  Protocol problem: connection reset
```

### （4）使用组件

- Logger
- Handler
- Filter
- Formatter

</br>


_logging.handlers_ 

日志处理程序，https://docs.python.org/zh-cn/3.9/library/logging.handlers.html#module-logging.handlers

- StreamHandler
- FileHandler
- NullHandler


demo1: 多个 handler

```python
import logging

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
```








### （5）异步日志操作

方式一：

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

方式二：


```python
import queue
from logging.handlers import QueueHandler, QueueListener

# instantiate queue & attach it to handler
log_queue = queue.Queue(-1)
queue_handler = QueueHandler(log_queue)

# instantiate our custom log handler (see question)
remote_handler = RemoteLogHandler()

# instantiate listener
remote_listener = QueueListener(log_queue, remote_handler)

# attach custom handler to root logger
logging.getLogger().addHandler(queue_handler)

# start the listener
remote_listener.start()

```

方法三（参考官方文档）：

```python
que = queue.Queue(-1)  # no limit on size
queue_handler = QueueHandler(que)
handler = logging.StreamHandler()

# use thread to dequeue
listener = QueueListener(que, handler)

root = logging.getLogger()
root.addHandler(queue_handler)
formatter = logging.Formatter('%(threadName)s: %(message)s')
handler.setFormatter(formatter)
listener.start()
# The log output will display the thread which generated
# the event (the main thread) rather than the internal
# thread which monitors the internal queue. This is what
# you want to happen.
root.warning('Look out!')
listener.stop()
```


</br>


### _queuehandler_

queuehandler 源码：


```python
class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.

    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def enqueue(self, record):
        """
        Enqueue a record.

        The base implementation uses put_nowait. You may want to override
        this method if you want to use blocking, timeouts or custom queue
        implementations.
        """
        self.queue.put_nowait(record)

    def prepare(self, record):
        """
        Prepares a record for queuing. The object returned by this method is
        enqueued.

        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.

        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        # The format operation gets traceback text into record.exc_text
        # (if there's exception data), and also returns the formatted
        # message. We can then use this to replace the original
        # msg + args, as these might be unpickleable. We also zap the
        # exc_info and exc_text attributes, as they are no longer
        # needed and, if not None, will typically not be pickleable.
        msg = self.format(record)
        # bpo-35726: make copy of record to avoid affecting other handlers in the chain.
        record = copy.copy(record)
        record.message = msg
        record.msg = msg
        record.args = None
        record.exc_info = None
        record.exc_text = None
        return record

    def emit(self, record):
        """
        Emit a record.

        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)
```



</br>


### _queuelisener_

queuelisener 源码：


```python
class QueueListener(object):
    """
    This class implements an internal threaded listener which watches for
    LogRecords being added to a queue, removes them and passes them to a
    list of handlers for processing.
    """
    _sentinel = None

    def __init__(self, queue, *handlers, respect_handler_level=False):
        """
        Initialise an instance with the specified queue and
        handlers.
        """
        self.queue = queue
        self.handlers = handlers
        self._thread = None
        self.respect_handler_level = respect_handler_level

    def dequeue(self, block):
        """
        Dequeue a record and return it, optionally blocking.

        The base implementation uses get. You may want to override this method
        if you want to use timeouts or work with custom queue implementations.
        """
        return self.queue.get(block)

    def start(self):
        """
        Start the listener.

        This starts up a background thread to monitor the queue for
        LogRecords to process.
        """
        self._thread = t = threading.Thread(target=self._monitor)
        t.daemon = True
        t.start()

    def prepare(self, record):
        """
        Prepare a record for handling.

        This method just returns the passed-in record. You may want to
        override this method if you need to do any custom marshalling or
        manipulation of the record before passing it to the handlers.
        """
        return record

    def handle(self, record):
        """
        Handle a record.

        This just loops through the handlers offering them the record
        to handle.
        """
        record = self.prepare(record)
        for handler in self.handlers:
            if not self.respect_handler_level:
                process = True
            else:
                process = record.levelno >= handler.level
            if process:
                handler.handle(record)

    def _monitor(self):
        """
        Monitor the queue for records, and ask the handler
        to deal with them.

        This method runs on a separate, internal thread.
        The thread will terminate if it sees a sentinel object in the queue.
        """
        q = self.queue
        has_task_done = hasattr(q, 'task_done')
        while True:
            try:
                record = self.dequeue(True)
                if record is self._sentinel:
                    if has_task_done:
                        q.task_done()
                    break
                self.handle(record)
                if has_task_done:
                    q.task_done()
            except queue.Empty:
                break

    def enqueue_sentinel(self):
        """
        This is used to enqueue the sentinel record.

        The base implementation uses put_nowait. You may want to override this
        method if you want to use timeouts or work with custom queue
        implementations.
        """
        self.queue.put_nowait(self._sentinel)

    def stop(self):
        """
        Stop the listener.

        This asks the thread to terminate, and then waits for it to do so.
        Note that if you don't call this before your application exits, there
        may be some records still left on the queue, which won't be processed.
        """
        self.enqueue_sentinel()
        self._thread.join()
        self._thread = None
```




---------

[日志教程](https://docs.python.org/zh-cn/3/howto/logging.html#logging-advanced-tutorial)

[日志进阶](https://docs.python.org/zh-cn/3/howto/logging-cookbook.html#logging-cookbook)，在多个模块使用日志






-------------
参考资料：
- [官方文档](https://docs.python.org/zh-cn/3.9/library/logging.html)
- [logging-cookbook](https://docs.python.org/zh-cn/3.9/howto/logging-cookbook.html#logging-cookbook)
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


设置输出的日志级别：
```python
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")
```





</br>

示例：

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231026231754.png">

</br>


更多语法参考：https://github.com/Delgan/loguru


-------------

异步日志好处：
- **提高性能**，将日志记录与主程序的执行分离，不会阻塞主程序的执行，特别是高并发或高负载的场景
- **减少 IO 开销**，异步日志操作可以将多个日志请求合并成一个批量写入操作，减少磁盘 IO 次数（磁盘IO 是一个相对很慢的操作）
- **简化代码**，便于维护，将日志记录逻辑与主程序的逻辑分离