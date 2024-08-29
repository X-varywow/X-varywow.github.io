
atexit, 退出处理器。

https://docs.python.org/zh-cn/3.12/library/atexit.html


程序退出时会执行这些函数：

```python
def goodbye(name, adjective):
    print('Goodbye %s, it was %s to meet you.' % (name, adjective))

import atexit

atexit.register(goodbye, 'Donny', 'nice')
# or:
atexit.register(goodbye, adjective='nice', name='Donny')
```

装饰器用法：

```python
import atexit

# 只有在函数不需要任何参数调用时才能工作
@atexit.register
def goodbye():
    print('You are now leaving the Python sector.')
```
