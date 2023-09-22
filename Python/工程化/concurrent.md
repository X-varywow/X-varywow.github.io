
利用多线程或多进程，来实现并发或并行。

python 中因为 GIL 的存在，所以一般使用多进程来实现。

不必过早使用并发并行，需要优化时，再根据 IO密集型 或 CPU密集型 选择合适的模块。

CPU 密集型，建议使用进程池，加速计算（大小不超过核心数）

IO 密集型，建议使用线程池，增加并发。同时，不要频繁地创建销毁进程、线程池。



## multiprocessing

参考：[官方文档](https://docs.python.org/zh-cn/3/library/multiprocessing.html)

>提供了本地和远程并发操作，通过 **使用子进程而非线程，有效地绕过了 全局解释器锁**


?>GIL 是一个防止多线程并发执行机器码的互斥锁，每个解释器进程都具有一个 GIL. </br>
</br>GIL 并不是Python的特性，它是在实现Python解析器(CPython)时所引入的一个概念。</br>
</br> 为什么 GIL 存在？</br>
（1）历史遗留原因</br>
（2）实现 CPython 在内存管理机制上的非线程安全 </br>
（3）单线程下可以保证较高的效率 </br>
（4）降低了集成 C 库的难度（避免考虑线程安全），促进了 C 库和 Python 的结合。</br>
</br> Python如何利用多核处理器？</br>
使用多进程而非多线程。每个进程拥有独立的解释器、GIL 以及数据资源，多个进程之间不会再受到 GIL 的限制。


（1）使用 Pool ⭐️

```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```

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
    p.join()
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


（4）更多

- Value
- Array
- Lock



## concurrent.futures

参考：[官方文档](https://docs.python.org/zh-cn/3/library/concurrent.futures.html)

>提供异步执行可调用对象高层接口。


常用例子：

_ThreadPoolExecutor_

Executor 的子类，它使用线程池来异步执行调用。

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

[如何在进程并发的时候使用 tqdm 进度条](https://gist.github.com/alexeygrigorev/79c97c1e9dd854562df9bbeea76fc5de)


## threading

基于线程的并发。

其实是假的多线程，不管有几个核，单位时间内只能跑一个线程，然后时间片轮转。

如果需要充分利用多核，可使用 multiprocessing 库，创建多进程。

----------

参考资料：
- [进程、线程、多线程、并发、并行 详解](https://cloud.tencent.com/developer/article/1744660)
- [谈谈python的GIL、多线程、多进程](https://zhuanlan.zhihu.com/p/20953544)
- [Python threading实现多线程 基础篇](https://zhuanlan.zhihu.com/p/91601448)
- [Python 并发编程（三）对比（multiprocessing, threading, concurrent.futures, asyncio）](https://blog.csdn.net/be5yond/article/details/120040690)
- [饱受争议的话题：全局解释器锁 GIL](https://juejin.cn/post/7121929253221826591)