
监控单个文件：

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 创建一个自定义事件处理器
class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            print(f"File {event.src_path} has been modified")

# 创建一个观察者并启动
observer = Observer()
event_handler = MyHandler()
observer.schedule(event_handler, path="path/to/your/file", recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
```





---------------

参考资料：
- https://zhuanlan.zhihu.com/p/680418251