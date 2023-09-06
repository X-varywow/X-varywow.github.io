

Python 是动态语言。在声明变量时，不需要显式声明数据类型。

但在工程化时，常常希望指定数据的类型，防止 BUG

--------------------------

（1）简单 demo

```python
# 指定类型
def greeting(name: str) -> str:
    return 'Hello ' + name
```

（2）类型别名

```python
# Vector 是个类型别名，刻互换
Vector = list[float]

def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]

# passes type checking; a list of floats qualifies as a Vector.
new_vector = scale(2.0, [1.0, -4.2, 5.4])
```

（3）使用复合注解

```python
from typing import List, Dict, Tuple


def mix(scores: List[int], ages: Dict[str, int]) -> Tuple[int, int]:
    return (0, 0)

# python 3.9 之后，将符合注解已经内置，不需要再 import
```


（4）使用 NewType 来创建不同的类型

```python
from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(524313)


def get_user_name(user_id: UserId) -> str:
    ...

# passes type checking
user_a = get_user_name(UserId(42351))

# fails type checking; an int is not a UserId
user_b = get_user_name(-1)
```

（5）使用联合类型 Union

```python
# Union[X, Y] 等价于 X | Y 

from typing import Union, List

U = Union[str, int]

def foo(a: U, b: U) -> List[U]:
    return [a, b]
```


-------------------

参考资料：
- [官方文档](https://docs.python.org/zh-cn/3/library/typing.html)
- [教程1](https://zhuanlan.zhihu.com/p/419955374)