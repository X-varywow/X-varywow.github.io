
>(1) 垃圾回收机制


python采用的是引用计数机制为主，标记-清除和分代收集两种机制为辅的策略。

主要是引用计数，当一个对象的引用计数归零，内存就直接释放了；但循环引用不适用这种方法，于是采用了“标记-清除”算法。

此外，通过“分代回收”以空间换时间的方法提高垃圾回收效率。

参考：[Python垃圾回收机制](https://zhuanlan.zhihu.com/p/83251959)

>(2) 多线程

Python 中多线程由于有GIL的存在，导致在任意时间内只有一个线程在运行，只有在处理IO密集型任务上多线程才能发挥实力。Python 中可以使用 threading 这个库完成多线程。

Python 的多线程在多核主机上十分鸡肋，于是使用 进程+协程 代替多线程的方式。

协程（是一种基于线程之上，但又比线程更加轻量级的存在）

参考：[多线程及GIL全局锁](https://www.cnblogs.com/hukey/p/7263207.html)


>(3) *args, **kwargs是什么意思?

*args: 可变位置参数。
*kwargs: 可变关键字参数。

>(4) 谈一谈Python中的装饰器

Python 中装饰器其实也是一种函数，它可以在不修改原函数代码的情况下扩展原函数功能。但是装饰器返回的是一个函数对象，装饰器利用了闭包的原理来实现。主要用于日志插入、权限管理等等。

>(5) 写一个时间装饰器

```python
import functools
import time

# functools.wraps 旨在消除装饰器对原函数造成的影响
def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print(func.__name__ + "time_cost:{}s".format(end_time-start_time))
        return res
    return clocked
```

>(6) 说明os、sys模块不同，并列举常用的模块方法

os 提供对使用操作系统函数的支持，sys 则提供有解释器访问或维护的变量以及解释器交互的一些函数。

```python
os.getcwd()
os.listdir()

os.walk() #生成指定目录下的文件夹以及文件
os.path.join()
```

```python
sys.path  #python 搜索模块时的路径
sys.argv  #命令行参数列表
sys.exit() #退出程序并返回指定的整数
```

>(7) 什么是lambda表达式？它有什么好处？

lambda 表达式是一种匿名函数，适用于一些简单功能的函数，不用为其命名。使语法更加简洁


>(8) Python里面如何拷贝一个对象？

分为浅拷贝和深拷贝，浅拷贝不会拷贝对象内部的子对象，仍属于原来的引用；而深拷贝会拷贝对象内部的子对象。且产生了一个新的对象。

浅拷贝使用 copy.copy()，深拷贝使用 copy.deepcopy()

>(9) Python 中的协程？

最开始时 Python 使用 yield 实现协程，3.5 之后引入了 async 和 await 来实现协程

[Python 中 异步协程 的 使用方法介绍](https://blog.51cto.com/csnd/5951495)

>(10) Python 中的异常机制？

Python 中异常也是一个对象，所有异常的基类都是 Exception。捕获异常可以使用 try...except...

当 try 中出现了异常就会转到 except 中执行。

关于assert?


>(11) Python中is和==的区别

is比较的是对象在内存的地址， ==比较的对象中的值

>(12) Python的自省?

type()，dir()，getattr()，hasattr()，isinstance()

vars() 参考：https://www.runoob.com/python/python-func-vars.html

>(13) 不可变数据类型和可变数据类型

- `不可变数据类型`：数值、字符串、元组
  - 改变值，变量的地址会改变
  - 值相同的两个变量，地址是一样的
- `可变数据类型`：列表，集合，字典
  - 改变值，变量的地址不会改变
  - 值相同的两个变量，地址是不同的

>(14) 类变量、实例变量

类变量，类的所有实例之间共享的值

实例变量，实例化之后，每个实例单独拥有的变量

>(15)` __new__` 和 `__init__` 的区别

__new__是在实例创建之前被调用的，因为它的任务就是创建实例然后返回该实例对象，是个静态方法。

__init__是当实例对象创建完成后被调用的，然后设置对象属性的一些初始值，通常用在初始化一个类实例的时候。是一个实例方法。

补充：静态方法属于整个类所有，调用它不需要实例化，可以直接调用；先new再init

>(16) Python 类方法

具体可分为类方法、实例方法和静态方法。

采用 @classmethod 修饰的方法为类方法；采用 @staticmethod 修饰的方法为静态方法；不用任何修改的方法为实例方法。


>(17) 单例设计模式

单例模式是一种常用的软件设计模式，该模式的主要目的是确保某一个类只有一个实例存在。

作为python的模块是天然的单例模式,

```python
#mysingleton.py
class Singleton(object):
    def foo(self):
        pass
singleton = Singleton()
```
```python
from mysingleton import singleton
singleton.foo()
```


参考：[Python中的单例模式的几种实现方式的及优化](https://www.cnblogs.com/huchong/p/8244279.html)




>(18) Python 中的堆

Python 中的堆使用 heapq 模块实现，

堆是一个二叉树，分为小根堆和大根堆，heapq 中的是小根堆，即最小的元素总是在根节点 heap[0]
