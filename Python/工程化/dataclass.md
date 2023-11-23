

_dataclasses_

Python 3.7 引入的一个装饰器，<u>用于简化类的创建和使用过程</u>。

自动生成一些常见的方法，如构造函数、属性访问器、repr() 和比较方法等。


---------


旧方法1, nametuple：

```python
from typing import NamedTuple
import sys

User = NamedTuple("User", [("name", str), ("surname", str), ("password", bytes)])

u = User("John", "Doe", b'tfeL+uD...\xd2')
print(f"Size: {sys.getsizeof(u)}")
```

旧方法2，定义对象属性：

```python
class Book:
    def __init__(self, name: str, price: float, author:str = "li"):
        self.name = name
        self.price = price
        self.author = author
```


---------------

新方法, dataclass：

```python
from dataclasses import dataclass, asdict, astuple

@dataclass
class Point:
    y: int
    x: int
    z: float = default_value

p = {"x": 3, "y": 1.1, "z":3.3}

# 利用**完成参数输入
Point(**p)

# 或利用 list 传参，顺序必须对上
a = Point(*list_name)

# 可以像访问普通属性一样访问和修改类的属性
a.x 
a.x = 3

# 将对象转化为 dict 并返回
asdict(a)
```


------------------


其它选项：
- frozen = True , 冻结类的实例，使其变为不可变类型
- order = True, 生成 `__lt__()`、`__le__()`、`__gt__()`、`__ge__()` 方法，使类的实例可以进行比较。


--------

pydantic 感觉是 dataclass 的一个改进：支持数据验证、序列化等

请参考左侧目录



------------------

参考资料：
- [官方文档](https://docs.python.org/zh-cn/3.11/library/dataclasses.html)
- chatgpt