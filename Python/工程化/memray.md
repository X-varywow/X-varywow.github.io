

分析应用内存占用

github stars: [memray](https://github.com/bloomberg/memray)(11.8K), [py-spy](https://github.com/benfred/py-spy)(11.4k), [scalene](https://github.com/plasma-umass/scalene)(10.6k) memory_profiler(4.1k)

## memray

```bash
pip install memray

# 实时命令行界面
memray run --live my_script.py

# 在另一个端口打开，本窗口用于观测程序日志
memray run --live-remote ./tests/test_copy.py
memray3.9 live 52614

# 查看分析日志
memray run tests/test_memory.py --aggregate

# summary 会在 bash 生成, 好看一些
memray summary out.bin 
memray stats out.bin
memray flamegraph out.bin
memray table out.bin
memray tree out.bin
```

但是 live 模式只能看内存个大概:
```python
import time
a = []
while 1:
    a += ['abcdefghijklmnopqrstuvwxyz'*1000]*100000
    time.sleep(1)
```
如监听上述文件，只能看到 \<module\> 的占用而看不到详细变量的占用;

代码内部比较细的内存占用，还得看火焰图


## memory_profiler

```python
# pip install memory_profiler

import numpy as np
from memory_profiler import profile
import time

@profile
def demo():
    for i in range(5):
        a = np.random.rand(1000000)
        b = np.random.rand(1000000)
        
        a_ = a[a < b]
        b_ = b[a < b]

        time.sleep(2000)
    
    del a, b

    return a_, b_


if __name__ == '__main__':
    demo()
```


之后测试代码：

```bash
python tests/test_memory.py

# 使用 mprof 查看内存随时间的变化图
mprof run tests/test_memory.py

mprof plot
```

参考资料：https://zhuanlan.zhihu.com/p/121003986


## tracemalloc

https://docs.python.org/zh-cn/3/library/tracemalloc.html


demo1: 显示内存分配最多的10行


```python
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

tracemalloc.start()

# ... run your application ...

snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
```

demo2: 获取快照并显示差异

```python
import tracemalloc
tracemalloc.start()
# ... start your application ...

snapshot1 = tracemalloc.take_snapshot()
# ... call the function leaking memory ...
snapshot2 = tracemalloc.take_snapshot()

top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("[ Top 10 differences ]")
for stat in top_stats[:10]:
    print(stat)
```

感觉跟踪不全啊，加载模型占用内存多了 1GB, 可这个只跟踪了10mb;

## objgraph

https://objgraph.readthedocs.io/en/stable/

```bash
pip install objgraph
```


```python
import objgraph

# 全局类型数量
objgraph.show_most_common_types(limit=50)

# 增量变化
objgraph.show_growth(limit=30)
```

## psutil

```python
import os
import psutil
from loguru import logger

def memory_query():
    process = psutil.Process()
    mem_info = process.memory_info()
    current_memory = mem_info.rss / (1024 * 1024)  # 转换为MB
    logger.info(f"Mem Usage: {current_memory:.2f} MB")
```

在服务中定时运行：

```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(memory_query, 'interval', seconds = 1)
scheduler.start()
```


## gc

https://docs.python.org/zh-cn/3/library/gc.html

```python
import gc
from loguru import logger

before = gc.get_count()
# your code 
after = gc.get_count()

logger.info(f"gc count: {before} -> {after}")
```

## sys.getsizeof

```python
import sys

# 创建一个包含1百万整数的列表
my_list = [1] * 1000000

# 使用getsizeof估算列表自身的内存占用
list_size = sys.getsizeof(my_list)

# 估算列表中所有整数的总内存占用
int_size = sys.getsizeof(1) * len(my_list)

# 总内存占用大约是列表自身的内存加上所有整数的内存
total_memory = list_size + int_size

print(f"列表自身的内存占用: {list_size} bytes")
print(f"列表中所有整数的总内存占用: {int_size} bytes")
print(f"总内存占用: {total_memory} bytes")
```



## 内存泄露常见场景

demo1. 变量使用不当

```python
global_list = []

def add_items():
    global global_list
    large_data = [1] * 1000000  # 创建一个大列表
    global_list.append(large_data)
    return large_data  # 每次调用都会增加引用，但不会减少

for _ in range(1000):
    add_items()  # 每次循环都会增加列表的大小，但不会释放
```


demo2. 循环引用

循环引用通常发生在两个或多个对象相互引用，并且没有外部引用指向它们的情况下。这会导致它们的引用计数永远不会达到0，因此垃圾收集器无法回收这些对象占用的内存。


```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

a = Node(1)
b = Node(2)
a.next = b
b.next = a  # 循环引用，无法被垃圾回收
```

demo3. 线程使用不当

```python
def thread_task():
    large_data = [1] * 1000000  # 线程局部变量，但未释放
    while True:
        pass

from threading import Thread
thread = Thread(target=thread_task)
thread.start()  # 线程持续运行，占用的内存无法释放
```

缓存数据

未关闭连接



## 记录一次内存泄露排查

step1. 收集信息

对比了泄露与不泄露的代码（没收获）

内存增长情况（与请求数量挂钩）

内存泄露常见case (循环引用，无限增长)

step2. 使用包

这个时候错误地估计了内存增长的量级，实际上内存增长很小，

然后模拟的量级不够，再加上这些包总是算不出 lgbm model 占用内存情况；所以认为这些包没啥用

step3. 手动更改代码

一步步更改代码，使用 psutil 记录全局的内存占用信息。和公司的监控平台；

量级、内存增长不稳定，导致没什么收获

step4. 增大量级

反应过来之前看的增长速度并不全是内存泄露带来的，

真实观测到内存泄露所需的量级较大；

使用 tracemalloc snapshot 抽离核心代码，大量请求观测 （1000 q）

找到可能原因（python pydantic basemodel: a, ）, 在 a 上加一个 [100]*1000000 属性； 发现不是，，，



