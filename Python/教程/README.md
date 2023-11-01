
欢迎来到 Python 教程


-----------

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