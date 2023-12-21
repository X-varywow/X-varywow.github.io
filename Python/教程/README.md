
欢迎来到 Python 教程

------------

一些皆是对象

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

-----------

使用自省等

type()，dir()，getattr()，hasattr()，isinstance()

```python
a = []

# 常用help
help(a)

# 查看源码等信息
a??

# 函数返回对象object的属性和属性值的字典对象
vars()

```


-----------

传参新方法


```python
param = {
    'uid': uid,
    'rid': rid,
}

def main(uid, rid):
    pass

main(**param)
```

-----------

**执行顺序**，python 默认从上到下顺序同步执行

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

