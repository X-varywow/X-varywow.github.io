
在python中方法名如果是__xxxx__()的，那么就有特殊的功能，叫做“魔法”方法


### init & del

`__init__` : 构造函数，在生成对象时调用

`__del__` : 析构函数，释放对象时使用

### repr & str

用于定义一个对象的字符串表示形式。

<u>str 目标是易于理解，repr 目标是准确一致</u>

```python
class Complex:
  
    # Constructor
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
  
    # For call to repr(). Prints object's information
    def __repr__(self):
        return 'Rational(%s, %s)' % (self.real, self.imag)    
  
    # For call to str(). Prints readable form
    def __str__(self):
        return '%s + i%s' % (self.real, self.imag)    

t = Complex(10, 20)

t               # use repr info
print(t)        # use str  info

print(str(t))   # Same as print(t)
# 10 + i20

print(repr(t))
# Rational(10, 20)
```

```python
# no __str__ and  no __repr__: 
print(t)

# <__main__.Complex object at 0x7f57b9639300>

# no __repr__:
a = {'a': t}
print(a)
# {'a': <__main__.Complex object at 0x7f57b963b2b0>}
```

repr 不能返回非字符串， str 可以

### item

`__setitem__` : 按照索引赋值

`__getitem__` : 按照索引获取值

### get set


`__get__` : 获取属性

`__set__` : 设置属性

`__delete__` : 删除属性


```python
class Descriptor:
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError("Value cannot be negative")
        instance._value = value

    def __get__(self, instance, owner):
        print(f"Getting value from {instance} of {owner}")
        return instance._value

# Descriptor 作为类属性被赋给 MyClass.attr，而不是实例属性
# 当访问 obj.attr 时，Python 首先检查 obj.__dict__ 中是否有 'attr' 这个键
# 自然触发了 __get__ 方法

class MyClass:
    attr = Descriptor()  # 类属性是一个描述符
    
    def __init__(self, value):
        self._value = value

obj = MyClass(42)
print(obj.attr)  # 会触发 __get__ 方法

obj.attr = 20    # 正常
obj.attr = -5    # 抛出 ValueError
```


使用场景：属性验证、延迟计算...




### len 

`__len__` : 获得长度

### cmp

`__cmp__` : 比较运算

### call

`__call__` : 函数调用

```python
class people:

    def __init__(self,n,a,w):   # 这是private函数
        self.name = n             #定义public成员变量。
        self.age = a
        self.__weight = w         #定义private成员变量。

    def set_weight(self , w):
        self.__weight = w

    def __call__(self):          
        print("%s 说: 我 %d 岁。我 %d kg。" %(self.name,self.age,self.__weight))

# 类中的 call() 函数
#   作用：类的实例对象可以作为一个函数去执行

xiaoming=people("小明",14,40)
xiaoming.set_weight(50)  

xiaoming()    #将一个实例对象作为函数
```

### calcu 

`__add__`: 加运算

`__sub__`: 减运算

`__mul__`: 乘运算

`__truediv__`: 除运算

`__mod__`: 求余运算

`__pow__`: 乘方

### new & init


__new__是在实例创建之前被调用的，因为它的任务就是创建实例然后返回该实例对象，是个静态方法。

__init__是当实例对象创建完成后被调用的，然后设置对象属性的一些初始值，通常用在初始化一个类实例的时候。是一个实例方法。


### all

`__all__` 是一个变量，用于指示模块中应该被导入的公共接口（变量、函数、类等）。 

当一个模块被导入时，Python 解释器会检查模块中的 `__all__` 变量，并且只导入其中列出的属性。如果 `__all__` 变量不存在或为空，则导入所有非下划线开头的属性。

使用 `__all__` 可以控制模块的命名空间，避免导入模块时引入不必要的属性，同时也可以提供模块的公共接口，方便其他程序员使用。


示例：出现在代码开头的导入说明

```python
from ... import ...

__all__ = ['available_backends', 'choose_backend', 'cdf', 'pdf', 'ppf']

def available_backends():
    pass
```



### import

import 用于动态地导入模块

\_\_import\_\_(name, globals=None, locals=None, fromlist=(), level=0)

```python
def available_backends():
    backends = [None]
    for backend in ['mpmath', 'scipy']:
        try:
            __import__(backend)
        except ImportError:
            continue
        backends.append(backend)
    return backends
```