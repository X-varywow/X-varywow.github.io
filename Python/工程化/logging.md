
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



-------------
参考资料：
- [官方文档](https://docs.python.org/zh-cn/3.9/library/logging.html)
- https://mp.weixin.qq.com/s/YZs8Ysa1z5LMRAlR0ajbww