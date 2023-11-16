

Python 3.7 引入的一个装饰器，用于简化类的创建和使用过程。

自动生成一些常见的方法，如构造函数、属性访问器、repr() 和比较方法等。


`dataclasses` 用来替代 `nametuple`


旧方法：

```python
from typing import NamedTuple
import sys

User = NamedTuple("User", [("name", str), ("surname", str), ("password", bytes)])

u = User("John", "Doe", b'tfeL+uD...\xd2')
print(f"Size: {sys.getsizeof(u)}")
```




新方法：

```python
from dataclasses import dataclass

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

```


------------------


其它选项：
- frozen = True , 冻结类的实例，使其变为不可变类型
- order = True, 生成 `__lt__()`、`__le__()`、`__gt__()`、`__ge__()` 方法，使类的实例可以进行比较。




------------------

参考资料：
- [官方文档](https://docs.python.org/zh-cn/3.11/library/dataclasses.html)
- chatgpt