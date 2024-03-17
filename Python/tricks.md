
## _一些皆是对象_

```python
import model

r = model.rating

# 后续直接使用 r() 执行
```

```python
def hello(name):
    print(f"hello, {name}")

def decorator_func():
    return hello

deco_hello = decorator_func()
deco_hello("Snow")

# 简易的装饰器思想
```

>一切事物(无论是数据结构如：数字、字符串、函数，还是代码结构如：函数、类、模块)都是对象;</br></br>
都可以像对象一样被赋值给变量，可以作为参数传递给函数，可以作为函数的返回值，也可以赋予属性或赋予方法。这就给Python的代码风格带来了极大的统一性和灵活性。



</br>

## _自省_

type()，dir()，getattr()，hasattr()，isinstance()

```python
a = []

# 常用help
help(a)

# 查看源码等信息
get_long??
?get_long

# 函数返回对象object的属性和属性值的字典对象
vars()

```


</br>

## _传参方法_


```python
param = {
    'uid': uid,
    'rid': rid,
}

def main(uid, rid):
    pass

main(**param)
```

</br>

## _异常处理_

捕获异常的目的：让程序能够优雅地处理错误情况，而不是直接抛出并停止运行。

通过异常捕获，可以让程序更加健壮和用户友好。

Python 中异常也是一个对象，<u>所有异常的基类都是 Exception</u>。

捕获异常可以使用 try...except... ; 当 try 中出现了异常就会转到 except 中执行。

```python
try:
    a = 1/0
except ZeroDivisionError:
    print("ZeroDivisionError")
except:
    pass
else:
    print("run with no except")
finally:
    print("run always")

print(1)
```

使用 traceback 打印堆栈

```python
import traceback

try:
    pass
except Exception:
    logger.error(f"monitor p7 error: {traceback.format_exc()}")
    # raise
```

使用 raise 会将捕获的异常抛出，然后程序退出（或交由上层处理）

raise 会保留原始的堆栈信息，而使用 raise e 可能会丢失原始的堆栈信息。

raise Exception(f"an error occurred while func,,,") 自定义异常信息




</br>

## _执行顺序_

python 默认从上到下顺序同步执行

```python
func1()
func2()

# func2() 将会在 func1() 结束后运行
```

使用多线程来应对并发，（许多web 框架会带有并发，可能引起 bug）

```python
import threading

t1 = threading.Thread(target=func1)
t2 = threading.Thread(target=func2)

# 几乎同时开始执行
t1.start()
t2.start()

# 等待线程结束
t1.join()
t2.join()
```


a.py 为 
```python
a = 1
def main():
    pass
```

b.py 为
```python
from a import main
```

a = 1 是作为一个 <u>顶层代码行</u>，当 python 解释器在执行 `from a import main` 的时候，会执行 a.py 中的所有顶层代码，这时候 a = 1 会被执行。



</br>

## _格式规范_

[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

autopep8

flake8

vscode 插件：black formatter 可以 format on save


</br>

## _分布式锁_

通常，工程项目一般部署在多个 POD 上，多个POD 现在需要维持一个全局的锁。

不保证同质数据只分发到一个 POD 上时，需要考虑外部存储缓存状态。即分布式锁


```python
import redis
from redis.lock import Lock

# 连接到Redis服务
client = redis.StrictRedis(host='redis-host', port=6379, db=0)

# 获取锁对象
lock = client.lock("my_lock_name", timeout=5)

# 尝试获取锁
if lock.acquire(blocking=False):
    try:
        # 执行需要同步的代码
        print("Lock acquired. Doing some work")
        # ... 在这里做一些工作 ...
    finally:
        # 释放锁
        lock.release()
else:
    print("Failed to acquire lock")
```

> 分布式锁：在分布式系统中保证多个计算节点之间进行互斥操作的一种同步机制。</br>
> 在服务，应用分布在不同服务器上时，当它们要访问或修改共享资源时，需要确保不会产生冲突，类似于单机系统

</br>

## _错误行为_

（1）将 lambda 定义 放在 for 中【影响性能】

```python
def main():
    for i in range(10):
        filter1 = lambda x: x%2==0
        if filter1(i):
            print(i)

main()
```

（2）默认值设置为可变数据类型【结果不可控】

```python
def f(a, L=[]):
    L.append(a)
    return L

print(f(1)) # -> [1]
print(f(2)) # -> [1, 2]
print(f(3)) # -> [1, 2, 3]
```

</br>

## _更多装饰器_


权限验证装饰器：

```python
def md5_decorator(level):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if st.session_state.auth_level >= level:
                return func(*args, **kwargs)
            else:
                return "权限不足"
        return wrapper
    return decorator

@md5_decorator(level = 1)
def main():
    pass
```

改成装饰器类：

```python
class MD5Decorator:
    def __init__(self, level):
        self.level = level

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if st.session_state.auth_level >= self.level:
                return func(*args, **kwargs)
            else:
                return "权限不足", 0
        return wrapper

@MD5Decorator(level=2)
def protected_function():
    # Function implementation
    pass
```




------------

参考资料：
- [python 官方文档：errors](https://docs.python.org/zh-cn/3/tutorial/errors.html)