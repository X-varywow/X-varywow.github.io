
## _一些皆是对象_

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

</br>

## _自省_

type()，dir()，getattr()，hasattr()，isinstance()

```python
a = []

# 常用help
help(a)

# 查看源码等信息
get_long??
?get_long

# 函数返回对象object的属性和属性值的字典对象
vars()

```


</br>

## _传参方法_


```python
param = {
    'uid': uid,
    'rid': rid,
}

def main(uid, rid):
    pass

main(**param)
```

</br>

## _异常处理_

捕获异常的目的：让程序能够优雅地处理错误情况，而不是直接抛出并停止运行。

通过异常捕获，可以让程序更加健壮和用户友好。

Python 中异常也是一个对象，<u>所有异常的基类都是 Exception</u>。

捕获异常可以使用 try...except... ; 当 try 中出现了异常就会转到 except 中执行。

```python
try:
    a = 1/0
except ZeroDivisionError:
    print("ZeroDivisionError")
except:
    pass
else:
    print("run with no except")
finally:
    print("run always")

print(1)
```

使用 traceback 打印堆栈

```python
import traceback

try:
    pass
except Exception:
    logger.error(f"monitor p7 error: {traceback.format_exc()}")
    # raise
```

使用 raise 会将捕获的异常抛出，然后程序退出（或交由上层处理）

raise 会保留原始的堆栈信息，而使用 raise e 可能会丢失原始的堆栈信息。




</br>

## _执行顺序_

python 默认从上到下顺序同步执行

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



</br>

## _格式规范_

[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

autopep8

flake8

vscode 插件：black formatter 可以 format on save


------------

参考资料：
- [python 官方文档：errors](https://docs.python.org/zh-cn/3/tutorial/errors.html)