

## 下划线

>函数使用单下划线_开头
>- 使用单下划线(_)开头的函数_func不能被模块外部以: from module import *形式导入。
>- 但可以用：from module import _func形式单独导入。

>类属性和类方法使用单下划线_开头
>- _开头为保护类型的属性和方法，仅允许类内部和子类访问，类实例无法访问此属性和方法。

>类属性和类方法使用双下划线__开头
>- __开头为私有类型属性和方法，仅允许类内部访问，类实例和派生类均不能访问此属性和方法。


\_\_init\_\_ 的作用：

```bash
--base_dir
    |--son_dir
        ||--__init__.py # 新增加的文件，可以为空，使son_dir变成可调用的package
        ||--module1.py
                def func1()
                def func2()
    |--module2.py
```

## 使用 raise, assert

```python
if n < 1:
    raise Exception('n must be at least 1')
```

## other

使用 with ，能够减少冗长，还能自动处理上下文环境产生的异常。


- 定义函数时，带默认参数的必须出现在参数列表最右边
- 多行注释`'''  '''`
- `eval()`执行字符串表达式
- `del list(index)`，删除
- `list`使用`+=`可进行扩充

```python
'' != None       # -> True
not ''           # -> True
```

```python
# nums[:]=ans 和 nums=ans.copy() 相比, 第二个高效些
```