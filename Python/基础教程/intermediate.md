本文基于 [Intermediate Python](https://readthedocs.org/projects/intermediatepythongithubio/downloads/pdf/latest/) 学习过程中摘要与实践。

书的更新日期：Jul 11, 2020.

> blueprint -(伪代码)</br>
> parentheses -(括号)</br>
> clause -(从句)

## _\*args and \*\*kwargs_


When you do not know beforehand how many arguments can be passed to your function,
use it:
```python
def test_var_args(f_arg,*argv):
    print("first normal arg:",f_arg)
    for arg in argv:
        print("another arg through *argv:",arg)
test_var_args('python','hello','wow')

#-->first normal arg: python
#-->another arg through *argv: hello
#-->another arg through *argv: wow
```
    
```python
def test_args_kwargs(arg1,arg2,arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3",  arg3)

kwargs={"arg3":3,"arg2":"two","arg1":5}
test_args_kwargs(**kwargs)

#-->arg1: 5
#-->arg2: two
#-->arg3 3
```

```python
def test_args_kwargs(**kwargs):
    print("arg1:", kwargs.get('arg1'))
    print("arg2:", kwargs.get('arg2'))
    print("arg3",  kwargs.get('arg3'))
```

传参的时候可以少写很多



`*args` is used to send a non-keyworded variable length argument list to the function.

`**kwargs` allows you to pass **keyworded** variable length of arguments to a function.

The **order** is `some_func(fargs, *args, **kwargs)`

The most common use case is when making function decorators.


## _Debugging_

You can run a script from the commandline using the **Python debugger**(pdb). Here is an example:
`$ python -m pdb my_script.py`
Running from inside a script:
```python
import pdb
def make_bread():
    pdb.set_trace()
    return  " i dont have time"
print(make_bread())
```

You would **enter the debugger pattern** as soon as you run it. Now it’s time to learn some of the commands of the debugger.
**Commands**:

- `c`: continue execution

- `w`: shows the context of the current line it is executing.
- `a`: print the argument list of the current function
- `s`: execute the current line and stop at the first possible occasion
- `n`: continue execution until the next line in the current function is reached or it returns


## _Generators_


According to Wikipedia, an iterator is an object that enables a programmer to traverse a container, particularly lists. 


<u>可迭代对象</u>, An **iterable** is any object in Python which has an `__iter__` or a `__getitem__` method
defined which returns an iterator or can take indexes.

<u>迭代器</u>, An **iterator** is any object in Python which has a  `__next__` method defined. 

<u>迭代</u>, When we use a loop to loop over something it is called **iteration**.

###### 生成器
- **Generators** are iterators, but you can only iterate over them once.
- It’s because they do **not store all the values in memory**, they generate the values on the fly.
- Most of the time generators are implemented as functions.

```python
def generator_function():
    for i in range(10):
        yield i
for item in generator_function():
    print(item)
```

```python
def generator_function():
	for i in range(2):
		yield i
gen = generator_function()
print(next(gen))  #-->0
print(next(gen))  #-->1
print(next(gen))  #-->error

# after yielding all the values next() caused a StopIteration error
```

```python
my_string = "Yasoob"
my_iter = iter(my_string)
next(my_iter)

# str是个可迭代对象，但不是迭代器
# 用iter()函数
```

## _Map, Filter and Reduce_


菜鸟教程链接：[map](https://www.runoob.com/python/python-func-map.html) [filter](https://www.runoob.com/python/python-func-filter.html) [reduce](https://www.runoob.com/python/python-func-reduce.html)


```python
items = [1, 2, 3, 4, 5]
squared = []
for i in items:
    squared.append(i**2)
#       ||
#       ||
#       \/
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
```

```python
def multiply(x):
    return (x*x)
def add(x):
    return (x+x)
funcs = [multiply, add]
for i in range(5):
    value = list(map(lambda x: x(i), funcs))

# use map even a list of functions
```

```python
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)

#-->[-5, -4, -3, -2, -1]

#If map & filter do not appear beautiful to you
# then you can read about list/dict/tuple comprehensions.(推导式)
```

```python
product = 1
list = [1, 2, 3, 4]
for num in list:
    product = product * num
#       ||
#       ||
#       \/
from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
```




## _Decorators_


Decorators are a significant part of Python. In simple words: they are functions which modify the functionality of other functions. They help to make our code shorter and more Pythonic.

```python
def hi(name="yasoob"):
    return "hi " + name
print(hi())
# output: 'hi yasoob'
# 我们甚⾄可以将⼀个函数赋值给⼀个变量，⽐如
greet = hi
# 我们这⾥没有在使⽤⼩括号，因为我们并不是在调⽤hi函数
# ⽽是在将它放在greet变量⾥头。我们尝试运⾏下这个
print(greet())
# output: 'hi yasoob'
# 如果我们删掉旧的hi函数，看看会发⽣什么！
del hi
print(hi())
#outputs: NameError
print(greet())
#outputs: 'hi yasoob'

# ======Everything in python is a object=====
```

```python
def hi(name="yasoob"):
    def greet():
        return "now you are in the greet() function"
    def welcome():
        return "now you are in the welcome() function"
    if name == "yasoob":
        return greet
    else:
        return welcome
a = hi()
print(a)
#-->function greet at 0x7f2143c01500>
#上⾯清晰地展示了`a`现在指向到hi()函数中的greet()函数
#现在试试这个
print(a())
#-->now you are in the greet() function

#=====在函数中定义函数，返回函数=====
```

```python
def hi():
    return "hi yasoob!"
def doSomethingBeforeHi(func):
    print("I am doing some boring work before executing hi()")
    print(func())
doSomethingBeforeHi(hi)
#-->I am doing some boring work before executing hi()
# hi yasoob!

#=====将函数作为参数传给另一个函数=====
```

```python
def a_new_decorator(a_func):
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
        a_func()
        print("I am doing some boring work after executing a_func()")
    return wrapTheFunction
def a_function_requiring_decoration():
    print("I am the function which needs some decoration")

a_function_requiring_decoration()
#outputs: "I am the function which needs some decoration"
a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
#now a_function_requiring_decoration is wrapped by wrapTheFunction()
a_function_requiring_decoration()
#outputs:I am doing some boring work before executing a_func()
# I am the function which needs some decoration
# I am doing some boring work after executing a_func()

###一个简单的装饰器，相当于赋给函数一个经修饰的函数
```

```python
@a_new_decorator
def a_function_requiring_decoration():
    print("I am the function which needs some decoration")
a_function_requiring_decoration()
#outputs: I am doing some boring work before executing a_func()
# I am the function which needs some decoration to remove my foul smell
# I am doing some boring work after executing a_func()

#重点理解一下 @ 符号

#但此时a_function_requiring_decoration.__name__变成了wrapTheFunction
#这时就有了functools.wraps，下面是蓝本：
```

```python
# blueprint
from functools import wraps
def decorator_name(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not can_run: 
            return "Function will not run"
        return f(*args, **kwargs)
    return decorated

@decorator_name
def func():
    return("Function is running")

can_run = True
print(func())
# Output: Function is running
can_run = False
print(func())
# Output: Function will not run

```

## _Mutation_

```python
foo = ['hi']
print(foo)
#-->['hi']

bar = foo
bar += ['bye']
print(foo)
#-->['hi', 'bye']

#将一个变量赋值为另一个可变类型变量时，对数据的改动会反映在两个变量上

#python可变类型（mutable）：列表，字典
#python不可变类型：数字，字符串，元组
```

```python
def add_to(num, target=[]):
    target.append(num)
    return target
add_to(1)
# Output: [1]
add_to(2)
# Output: [1, 2]
add_to(3)
# Output: [1, 2, 3]

#在Python中当函数被定义时，默认参数只会运算⼀次
```

## _Enumerate_


```python
my_list = ['apple', 'banana', 'grapes', 'pear']
for index,name in enumerate(my_list,1):
    print index,name

# Output:
# 1 apple
# 2 banana
# 3 grapes
# 4 pear
```

## _Object introspection_


```python
dir(object)
# It returns a list of attributes and methods belonging to an object

type(object)
# It returns the type of an object.

id(object)
# It returns the unique ids of various objects.
```

## _异常处理_

```python
try:
    file = open('test.txt', 'rb')
except Exception:
    # 打印⼀些异常⽇志，如果你想要的话
    raise

# try/else 在try不出现异常时触发
```

## _Classes_

```python
class Cal(object):
    # pi is a class variable
    pi = 3.142
    def __init__(self, radius):
        # self.radius is an instance variable
        self.radius = radius
    def area(self):
        return self.pi * (self.radius ** 2)

a = Cal(32)
a.area()
# Output: 3217.408
a.pi
# Output: 3.142
a.pi = 43
a.pi
# Output: 43

b = Cal(44)
b.area()
# Output: 6082.912
b.pi
# Output: 3.142
b.pi = 50
b.pi
# Output: 50
```

## _Function caching_

```python
from functools import lru_cache
@lru_cache(maxsize=32)
def fib(n):
    if n<2:return n
    return fib(n-1)+fib(n-2)
print([fib(n) for n in range(40)])

fib.cache_clear()


#这是一个被相同参数频繁调用的I/O密集的函数
#使用lru_cache可以使这段代码的运行时间从53.6s变为0s(笔者自测)
```

## 结语

本文漏掉了`set结构` `Ternary Operators(三元运算符）` `装饰器的高级应用` 
`Global & Return`(尽量return多值代替global)
`__slots__魔法` `Virtual Environment` `Collections`
`推导式` `lambda 参数:操作` `Python C extensions`
`for/else`(else会在for循环正常结束时执行)
`open函数` `Targeting Python 2+3` `Coroutines(协程)`
`上下文管理器`
