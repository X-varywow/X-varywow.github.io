
_字符串格式化_

```python
# 方法一：使用 %
print("我叫 %s 今年 %d 岁!" % ('小明', 10))

# 方法二：使用 format
print("我叫 {} 今年 {} 岁!".format('小明', 10))

# 方法三： 使用 f-string
print(f"my lucky number is {number}")
```

使用特殊方法：

```python
s.center(40, "#")

s.ljust(40, "#")

s.rjust(40, "#")
```


</br>

_format_


```python
"{} {}". format("hello", "world") # 不设置位置，默认排序

"{1} {0} {1}".format("hello", "world")  # 设置指定位置

"网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com") # 设置参数
```

```python
"网站名：{name}, 地址 {url}".format(**d) # 通过字典设置参数 (实用， d 可以是所需参数的超集)

"{a}".format_map(d)
```



| 格式      | 说明                                                      |
| --------- | --------------------------------------------------------- |
| `{:.2f}`  | 保留两位小数                                              |
| `{:-^70}` | 内容居中，使其70宽度，并用 - 填充；`<` 左对齐；`>` 右对齐 |
| `{:.2%}`  | 百分数格式                                                |



</br>

_f-string_

>f-string 在功能方面不逊于传统的 %-formatting 语句和 str.format() 函数，同时性能又优于二者，且使用起来也更加简洁明了。
</br>对于Python3.6及以后的版本，推荐使用 f-string 进行字符串格式化。


eg1:
```python
number = 7

f"my lucky number is {number}"

# -> my lucky number is 7
```

eg2:
```python
name = "JOHN"

f"my name is {name.lower()}"

# -> my name is john
```

eg3: 
```python
# f-string采用 {content:format} 设置字符串格式

a = 123.456

f"a is {a:.2f}"

# -> a is 123.46
```

eg4:
```python
# 先输出 k=, 再输出变量的值
# from None 表示异常链的起点就是这里，与不写一致；from other_exception 会引用异常链

raise TypeError(
    f'The number of choices must be a keyword argument: {k=}'
) from None
```


--------------

参考资料：
- [Python格式化字符串f-string概览](https://blog.csdn.net/sunxb10/article/details/81036693)
- 菜鸟教程