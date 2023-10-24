
在python中方法名如果是__xxxx__()的，那么就有特殊的功能，叫做“魔法”方法


### init & del

`__init__` : 构造函数，在生成对象时调用

`__del__` : 析构函数，释放对象时使用

### repr & str

`__repr__` & `__str__`

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
  
  
# Driver program to test above
t = Complex(10, 20)
  
# Same as "print t"
print (str(t))  
print (repr(t))
```

```python
class Cat:

    def __init__(self, new_name, new_age):
        self.name = new_name
        self.age = new_age  # 它是一个对象中的属性,在对象中存储,即只要这个对象还存在,那么这个变量就可以使用
        # num = 100  # 它是一个局部变量,当这个函数执行完之后,这个变量的空间就没有了,因此其他方法不能使用这个变量
 
    def __str__(self):
        return "名字是:%s , 年龄是:%d" % (self.name, self.age)
 
# 创建了一个对象
tom = Cat("汤姆", 30)
print(tom)
```

### item

`__setitem__` : 按照索引赋值

`__getitem__` : 按照索引获取值

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