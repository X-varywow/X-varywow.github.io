
面向对象（关注问题的关系、关注封装继承多态），面向过程（关注将问题拆分为一系列的解决步骤），都是编程范式的一种。

其它范式：面向切面编程（AOP, 关注模块化）、反应式编程（事件驱动，循环从事件队列取数）、

## 基本使用

### 1.0 定义与调用

```python
class MyClassA:
    i = 12345               # 类属性
    
    def funcA(self):        # 2. 类的函数需要显式指明self参数
        i = 10
        self.j = 2345         # 实例属性
        
        print("函数的局部变量i：", i)
        print("类属性i：", self.i)

obj = MyClassA()  # 定义类的对象，对类实例化
obj.funcA()
obj.i             # 类里的变量和函数，默认公开访问
MyClassA.i        # 类属性多了一种访问方式
```

类属性是所有实例共享的属性，改变一个会影响所有的。



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


demo1:

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


demo2: 自定义配置类（继承自 dict）

```python
import json

config = {}
with open("config.json") as f:
    config = json.load(f)


class DynamicConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)


config = DynamicConfig(config)

# 如果不设定 self.__dict__ = self 会为 {}
config.__dict__
```


!> 如果继承的父类 `__init__` 不需要接受除了 self 之外的参数，`super().__init__()` 可以省略


demo: lgbm 中多重继承

```sql
class LGBMRegressor(_LGBMRegressorBase, LGBMModel):
    """LightGBM regressor."""

    def fit(  # type: ignore[override]
        self,
        X: _LGBM_ScikitMatrixLike,
        y: _LGBM_LabelType,
        sample_weight: Optional[_LGBM_WeightType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        eval_set: Optional[List[_LGBM_ScikitValidSet]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_LGBM_WeightType]] = None,
        eval_init_score: Optional[List[_LGBM_InitScoreType]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        feature_name: _LGBM_FeatureNameConfiguration = 'auto',
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = 'auto',
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[str, Path, Booster, LGBMModel]] = None
    ) -> "LGBMRegressor":
        """Docstring is inherited from the LGBMModel."""
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model
        )
        return self

    _base_doc = LGBMModel.fit.__doc__.replace("self : LGBMModel", "self : LGBMRegressor")  # type: ignore
    _base_doc = (_base_doc[:_base_doc.find('group :')]  # type: ignore
                 + _base_doc[_base_doc.find('eval_set :'):])  # type: ignore
    _base_doc = (_base_doc[:_base_doc.find('eval_class_weight :')]
                 + _base_doc[_base_doc.find('eval_init_score :'):])
    fit.__doc__ = (_base_doc[:_base_doc.find('eval_group :')]
                   + _base_doc[_base_doc.find('eval_metric :'):])
```




### 1.4 组合

```python
class Engine:
    def __init__(self, displacement):
        self.displacement = displacement

    def start(self):
        print('引擎启动，排量为 {} 升.'.format(self.displacement))

class Car:
    def __init__(self, brand, engine):
        self.brand = brand
        self.engine = engine

    def start_car(self):
        print('{} 汽车准备启动.'.format(self.brand))
        self.engine.start()

# 创建一个引擎实例
my_engine = Engine(2.5)

# 创建一个搭载了这个引擎的汽车实例
my_car = Car('宝马', my_engine)

# 启动汽车
my_car.start_car()
```


关键的 OOP 原则：

|      |                                                                                  |
| ---- | -------------------------------------------------------------------------------- |
| 封装 | 将数据和方法捆绑一起，**隐藏内部状态和实现细节，只通过公共接口暴露对象的行为**。 |
| 继承 |                                                                                  |
| 多态 | 重写父类方法、鸭子类型                                                           |
| 抽象 |                                                                                  |
| 组合 | 将一个类的实例作为另一个类的属性                                                 |


- 接口隔离（不应该强迫客户依赖于它们不使用的接口）
- 单一职责（类的设计应该集中在一个核心功能上，而不是试图解决多个不相关的问题）
- 开闭原则（开闭原则指出软件实体（类、模块、函数等）应该对扩展开放，对修改关闭）



## 高级使用


### 2.0 抽象基类

Abstract Base Classes（抽象基类）, 不能被实例化，可以作为其它类的基类。


```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    
    @abstractmethod
    def fetch_data(self):
        """
        子类必须实现此方法来获取数据。
        """
        pass
```

任何继承的子类都需要实现 fetch_data 方法，不然还是抽象类

keyword: abc abc.ABC 强制实现 代码复用 接口定义



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

## 装饰器


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




---------------------

参考资料：
- 学校课程
- [实例讲解Python中的魔法函数（高级语法）](https://zhuanlan.zhihu.com/p/344951719)
- [runoob python oop](https://www.runoob.com/python/python-object.html)
