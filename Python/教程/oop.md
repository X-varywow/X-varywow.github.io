基本使用
---------------

### 1.0 定义与调用

```python
class MyClassA:
    i = 12345               # 1. 这里定义了类变量
    
    def funcA(self):        # 2. 类的函数需要显式指明self参数
        i=10
        self.j=2345         # 3. 这里通过self来区分局部变量和实例变量。 
        
        print("函数的局部变量i：",i)
        print("类属性i：",self.i)

obj = MyClassA()  #定义类的对象，对类实例化
obj.funcA()
obj.i             #类里的变量和函数，默认公开访问
```

```python
class foo(object):
    def f():             #相当于类外的，用对象调用时会默认带一个self参数
        print('hhh')

a=foo()
a.f()
```
--> `TypeError: f() takes 0 positional arguments but 1 was given`
f()定义时加个参数self即可运行


```python
class a:
    def __init__(self, b):
        self.b = b
        self.update()
    def update(self):
        self.b =3
        
model = a(2)
model.b

# --> 3
```


### 1.1 构造函数

```python
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart    #实例属性
        self.i = imagpart
                
            
x = Complex(3.0, -4.5)   #按照__init__()指定的参数构造对象
print(x.r, x.i)   # 输出结果：3.0 -4.5

# 可通过 x.r = 9 修改

# python 同样设计了析构函数：__del__()
```

### 1.2 私有属性

自定义的私有变量或函数，加 `__` 前缀。

```python
class MyClassC:
    attrA = 333         # 类属性。等同于static属性
    __attrB = 444
       
    def __init__(self, a, b, c, d):
        
        attrA=10         #局部变量，临时变量
        
        self.attrC  =c     #定义两个实例属性
        self.__attrD=d     #定义private成员变量

    @property
    def attrD(self):
        return self.__attrD 
                
    def show(self):
        print(self.attrA)
        print(self.__attrB)
        print(self.attrC)
        print(self.__attrD)
        
        
    def funcA(self,a,b):
        self.attrA=a       #注意这里！
        self.__attrB=b     #对象定义了两个新的实例属性，会覆盖掉同名的类属性，但另一个对象中未变

# 如果类属性是公开的，可以直接用类名字来调用
print(MyClassC.attrA)
```

[python类内部成员的访问及外部访问](https://blog.csdn.net/weixin_43984760/article/details/88358006)


### 1.3 继承

```python
#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))

#单继承示例
class student(people):
    grade = ''
    def __init__(self,n,a,w,g):
        #调用父类的构函
        people.__init__(self,n,a,w)
        self.grade = g
    #覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))

s = student('ken',10,60,3)
s.speak()
```

高级使用
-------------

### 2.0 抽象类
- 接口：若干抽象方法的集合。
  - 作用：限制实现接口的类必须按照给定的调用方式实现这些方法，对高层模块隐藏了类的内部实现。

```python
from abc import ABCMeta, abstractmethod

class Payment(metaclass = ABCMeta):
    @abstractmethod
    def pay(self, money):
        pass

# 父类，通过定义函数接口，实现抽象类
# 子类实现该接口，不然还是抽象类，无法实例化
```

### 2.1 静态方法

##### @staticmethod

静态方法，不需要表示自身对象的 `self` 和自身类的 `cls` 参数，就跟使用函数一样。

使用场景：不需要用到与类相关的属性或方法时

```python
class foo(object):
    @staticmethod                #返回函数的静态方法
    def f():
        print('hhh')

foo.f()   #静态方法无需实例化
a=foo()
a.f()     #也可以实例化后调用
```

### 2.2 类方法
##### @classmethod

类方法， 不需要 `self` 参数，但第一个参数需要时表示自身类的 `cls` 参数。

使用场景：需要用到与类相关的属性或方法，然后又想表明这个方法是整个类通用的，而不是对象特异的。

```python
class MyClassC:
    attrA = 333         # 类属性。等同于static属性

    @classmethon         # 类方法的装饰器。下面定义的函数为类方法
    def set_attrA(cls,a):   # 类方法的第一个参数不再是self，而是cls。
        cls.attrA=a      #相应，不再用self来引导，而是用cls  
```

### 2.3 运算符重载
```python
class Vector:
   def __init__(self, a, b):
      self.a = a
      self.b = b

   def __str__(self):
      return 'Vector (%d, %d)' % (self.a, self.b)

   def __add__(self,other):
      return Vector(self.a + other.a, self.b + other.b)

v1 = Vector(2,10)
v2 = Vector(5,-2)
print (v1 + v2)
```

装饰器
--------------------

###  3.1 内置装饰器
##### @property

该装饰器，将方法包装成属性，让方法可以以属性的形式被访问和调用。

```python
class foo:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x

#被装饰的方法不能传递除self外的其它参数

a = foo()
a.x = 6          #设置属性
print(a.x)       #获取属性
del a.x          #删除属性
```
- 如果报错 `RecursionError: maximum recursion depth exceeded while calling a Python object`，很可能是对象属性名和 `@property` 装饰的方法名重名了，一般会在对象属性名前加一个下划线 `_` 避免重名，并且表明这是一个受保护的属性。

### 3.2 一段实例
餐厅_装饰模式:
```python
class Food:
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost
    def get_name(self):
        return self.name
    def get_cost(self):
        return self.cost

class Crepe(Food):
    def __init__(self, name, cost):
        super().__init__(name, cost)  
    def make_base(self):
        print("Crepe: make_baseline")
    def add_options(self):
        print("Crepe: add_options")
    def make_it(self):
        self.make_base()
        self.add_options()

class Handcake(Food):
    def __init__(self, name, cost):
        super().__init__(name, cost)  
    def make_base(self):
        print("Handcake: make_baseline")
    def add_options(self):
        print("Handcake: add_options")
    def make_it(self):
        self.make_base()
        self.add_options()


class Egg(Food):
    def __init__(self, name, cost):
        super().__init__(name, cost)  
    def make_base(self):
        print("Egg: make_baseline")
    def add_options(self):
        print("Egg: add_options")
    def make_it(self):
        self.make_base()
        self.add_options()

def cold(c):
    def add_cold():
        print("add_cold")
    def wrapper():
        c.make_it()
        add_cold()
        print("{} costs {} $".format(c.get_name(), c.get_cost()))
    return wrapper()         


def hot(c):
    def add_pepper():
        print("add_pepper")
    def wrapper():
        c.make_it()
        add_pepper()
        print("{} costs {} $".format(c.get_name(), c.get_cost()))
    return wrapper()         #返回一个执行的函数

c = Crepe("Crepe", 10)
c.make_it()

print("\nAfter decorate：")
hot(c)
```


## 魔法方法

在python中方法名如果是__xxxx__()的，那么就有特殊的功能，因此叫做“魔法”方法


(1) 

`__init__` : 构造函数，在生成对象时调用

`__del__` : 析构函数，释放对象时使用

(2) 

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

(3) 

`__setitem__` : 按照索引赋值

`__getitem__` : 按照索引获取值

(4) 

`__len__` : 获得长度

(5) 

`__cmp__` : 比较运算

(6) 

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

(7) 

`__add__`: 加运算

`__sub__`: 减运算

`__mul__`: 乘运算

`__truediv__`: 除运算

`__mod__`: 求余运算

`__pow__`: 乘方

(8) 

`__new__` & `__init__`

__new__是在实例创建之前被调用的，因为它的任务就是创建实例然后返回该实例对象，是个静态方法。

__init__是当实例对象创建完成后被调用的，然后设置对象属性的一些初始值，通常用在初始化一个类实例的时候。是一个实例方法。


(9)

`__all__` 是一个变量，用于指示模块中应该被导入的公共接口（变量、函数、类等）。 

当一个模块被导入时，Python 解释器会检查模块中的 `__all__` 变量，并且只导入其中列出的属性。如果 `__all__` 变量不存在或为空，则导入所有非下划线开头的属性。

使用 `__all__` 可以控制模块的命名空间，避免导入模块时引入不必要的属性，同时也可以提供模块的公共接口，方便其他程序员使用。






---------------------

参考资料：
- 学校课程
- [实例讲解Python中的魔法函数（高级语法）](https://zhuanlan.zhihu.com/p/344951719)
