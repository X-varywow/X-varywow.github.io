
利用多线程或多进程，来实现并发或并行。

不必过早使用并发并行，需要优化时，再根据 IO密集型 或 CPU密集型 选择合适的模块。

---------

对于 CPU 密集型，建议使用进程池，并行加速计算（大小不超过核心数）（multiprocessing 和 ProcessPoolExecutor 都行）

对于 IO 密集型，建议使用**开销较少**的线程池，增加并发（如 concurrent.futures 中的 ThreadPoolExecutor）。

同时，不要频繁地创建销毁进程、线程池。


-----------


?>GIL 是一个防止多线程并行执行机器码的互斥锁，每个解释器进程都具有一个 GIL. </br>
</br>GIL 并不是Python的特性，它是在实现Python解析器(CPython)时所引入的一个概念。</br>
</br> 为什么 GIL 存在？</br>
（1）历史遗留原因</br>
（2）实现 CPython 在内存管理机制上的非线程安全 </br>
（3）单线程下可以保证较高的效率 </br>
（4）降低了集成 C 库的难度（避免考虑线程安全），促进了 C 库和 Python 的结合。</br>
</br> Python如何利用多核处理器？</br>
使用多进程而非多线程。每个进程拥有独立的解释器、GIL 以及数据资源，多个进程之间不会再受到 GIL 的限制。

对于 IO 密集型任务，即使有 GIL全局解释器锁（线程锁， 确保任何时刻只有一个线程执行 python 字节码），线程池也是有效的； 因为在执行 IO 操作的时候，线程可能处于等待状态，这时 GIL 会释放允许其他线程运行。

**GIL 更多的是在多核上多线程（CPU密集型任务）效率不行，把资源锁成1核的**


</br>

## multiprocessing

参考：[官方文档](https://docs.python.org/zh-cn/3/library/multiprocessing.html)

提供了本地和远程并发操作，通过 使用子进程而非线程，有效地绕过了 全局解释器锁


（1）使用 Pool ⭐️

```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```

----------


每个子进程都有自己的内存空间，当在子进程中修改全局变量时，只会影响到自己内存空间中的变量副本，不会影响主进程中的原始变量。

可以使用如下方式：

```python
from multiprocessing import Pool, Value

def f(x, cnt):
    cnt.value += 1
    print(cnt.value)

if __name__ == '__main__':
    cnt = Value('i', 0)  # 'i' 表示整数类型
    with Pool(5) as p:
        p.starmap(f, [(1, cnt), (2, cnt), (3, cnt), (4, cnt), (5, cnt)])
    print(cnt.value)
```




-----------

p.map(func, iterable) 对于很长的迭代独享，会消耗很多内存。可以考虑使用 imap() 或 imap_unordered() 并且指定 chunksize 以提升效率。

```python
from multiprocessing import Pool
from tqdm import tqdm


def f(x):
    return x * x


if __name__ == '__main__':
    with Pool(5) as p:
        print(list((tqdm(p.imap(f, range(10)), total=10, desc='监视进度'))))

```



（2）Process

```python
from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join() # 等待进程 p 完成执行
```

```python
from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
```



（3）进程之间交换对象

- Queue
- Pipe


```python
# 使用队列在进程之间通信
# 队列是线程安全，进程安全的

from multiprocessing import Process, Queue

def f(q):
    q.put(["some info"])

if __name__ == "__main__":
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())
    p.join()
```



（4）更多

- Value
- Array
- Lock

```python
# 使用同步锁

from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello', i)
    finally:
        l.release()

if __name__ == "__main__":
    lock = Lock()
    for num in range(10):
        Process(target=f, args=(lock, num)).start()

```




</br>

## concurrent.futures

参考：[官方文档](https://docs.python.org/zh-cn/3/library/concurrent.futures.html)

提供异步执行可调用对象高层接口；

由 ThreadPoolExecutor 使用线程或由 ProcessPoolExecutor 使用单独的进程来实现异步执行。

</br>

_ThreadPoolExecutor_

Executor 的子类，它使用线程池来异步执行调用。

demo1:

```python
from concurrent.futures import ThreadPoolExecutor
thread_pool = ThreadPoolExecutor(20)
thread_pool.submit(func, *params)
```



demo2:

```python
import concurrent.futures
import urllib.request

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://nonexistant-subdomain.python.org/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))
```


对于次序敏感的场景，注意并行任务合并结果时，由于 **处理时间导致错位** 的问题，即下述 res 中的顺序大概率是错的：

```python
def main(data):
    res = []
    with ThreadPoolExecutor(max_workers = 5) as executor:
        futures = [executor.submit(your_func, chunk) for chunk in np.array_split(data.values, 10)]
        for future in tqdm(as_completed(futures), total = len(futures)):
            res.extend(future.result())
    return res
```




</br>

_ProcessPoolExecutor_

Executor 的子类，它使用进程池来异步地执行调用。

基于 multiprocessing 模块，这允许绕过 GIL

```python
import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()
```


----------

demo: 使用线程池异步执行 dynamo 的更新

```python
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

def wrapper():
    dynamo.update(87654321, 12343345, 30)
    

executor = ThreadPoolExecutor(max_workers=3)

futures = [executor.submit(dynamo.update, 87654321, 12343345, 30) for i in range(200)]

wait(futures, return_when=ALL_COMPLETED, timeout=3)
```




[如何在进程并发的时候使用 tqdm 进度条](https://gist.github.com/alexeygrigorev/79c97c1e9dd854562df9bbeea76fc5de)

实例：[psycopg 连接池+多线程查询](/Python/web/psycopg?id=多线程amp连接池)


奇怪，各种超时设置都不能主动退出

```python
%%time
from concurrent.futures import ALL_COMPLETED
from tenacity import retry, stop_after_delay, stop_after_attempt, wait
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

STOP_AFTER_DELAY = 0.2  # 单位为秒
STOP_AFTER_ATTEMPT = 2  # 尝试次数
TIMEOUT_EXECUTION_IN_SECONDS = 1

def test():
    for i in range(999999999):
        for j in range(999999999):
            a = i*j
    return 1

@retry(stop=(stop_after_delay(STOP_AFTER_DELAY) | stop_after_attempt(STOP_AFTER_ATTEMPT)))
def multiprocess_demo():
    res = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(test) for _ in range(10)]
        done, _ = wait(futures, return_when=ALL_COMPLETED, timeout=TIMEOUT_EXECUTION_IN_SECONDS)
        try:
            for future in done:
                result = future.result()
                if result:
                    res += result
        except TimeoutError as e:
            print("Task timed out")
            raise e
        except Exception as e:
            print(f"Task failed: {e}")
    return res

print(multiprocess_demo())
```





</br>

## threading

基于线程的并发。

其实是假的多线程，不管有几个核，单位时间内只能跑一个线程，然后时间片轮转。

如果需要充分利用多核，可使用 multiprocessing 库，创建多进程。

```python
import threading

threading.Thread(target=func, daemon=True).start()
```

?> daemon 表示一个进程是否是守护进程；主线程中创建的线程默认为 daemon = False </br>
守护线程，在主程序退出时不会等待守护线程的完全，相当于一个优先级低的线程，常用于：程序自检、清理。</br>
存在普通线程未完成时，程序会一直执行，无法退出。


----------

threading.Event()

线程间通信的同步原语

- set(), 将内部标志设置为 True, 唤醒所有等待的线程
- clear()，内部标志重置为 False, 等待的会被阻塞
- is_set() 














----------

参考资料：
- [进程、线程、多线程、并发、并行 详解](https://cloud.tencent.com/developer/article/1744660)
- [谈谈python的GIL、多线程、多进程](https://zhuanlan.zhihu.com/p/20953544)
- [Python threading实现多线程 基础篇](https://zhuanlan.zhihu.com/p/91601448)
- [Python 并发编程（三）对比（multiprocessing, threading, concurrent.futures, asyncio）](https://blog.csdn.net/be5yond/article/details/120040690)
- [饱受争议的话题：全局解释器锁 GIL](https://juejin.cn/post/7121929253221826591)