
> 正则匹配，极强的字符匹配、提取工具

## _re函数_

`re.match(pattern,str,flags=0)`
从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。

`re.search()` 
扫描整个字符串并返回第一个成功的匹配。

`re.sub(pattern, repl, string, count=0, flags=0)`
替换字符串中的匹配项。

`re.compile()` 
用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用。

`re.findall()`
在字符串中找到正则表达式所匹配的所有子串，并**返回一个列表**，如果没有找到匹配的，则返回空列表。

```python
import re
print(re.match("h","hello"))
print(re.match("el","hello"))
print(re.match("h","hello").span())

#--> <re.Match object; span=(0, 1), match='h'>
#--> None
#--> (0,1)
```

</br>

## _正则符号_

`[ ]`
匹配需要的字符**集合**

`( )`
捕获需要的字符。

`(?: )`
非捕获分组。

`^`：脱字符号。
**方括号中加入**脱字符号，就是匹配未列出的所有其他字符，如`[^a]`匹配除a以外的所有其他字符。

`\`
和python字符串使用规则一样，可以匹配特殊字符本身。
如`\d`表示匹配0到9的任意一个数字字符，而`\\d`则表示匹配`\d`本身。

`|`
相当于或

----------------------------

多次匹配
- `*` : 匹配前一个字符0到n次，如pytho*n可以匹配pythn、pytoon、pythooooon等。
- `?` : 匹配前一个字符0或1次。
- `+` : 匹配前一个字符1到n次。等价于`{1,}`
- `{n,m}` : 匹配前一个字符n次到m次。
- `{n}`: 匹配前一个字符n次。

-----------------------------


| 模式 | 描述                                                                                     |
| ---- | ---------------------------------------------------------------------------------------- |
| `^`  | 匹配字符串的开头                                                                         |
| `$`  | 匹配字符串的末尾。                                                                       |
| `.`  | 匹配任意字符，除了换行符                                                                 |
| `\s` | 匹配任意空白字符，等价于 [\t\n\r\f]。                                                    |
| `\S` | 匹配任意非空字符                                                                         |
| `\d` | 匹配任意数字，等价于 [0-9]。                                                             |
| `\D` | 匹配任意非数字                                                                           |
| `\w` | 匹配数字字母下划线                                                                       |
| `\W` | 匹配非数字字母下划线                                                                     |
| `\A` | 匹配字符串开始                                                                           |
| `\z` | 匹配字符串结束                                                                           |
| `\b` | 匹配一个单词边界。例如， 'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。 |

正则匹配修饰符`re.S`使 `.` 匹配包括换行在内的所有字符

</br>

## _Demo_



demo1: 捕获 sql 中的 schema.table

```python
import re

pattern = r"(?i)from (\w+)\.(\w+)"
try:
    match = re.search(pattern, "select * from s.t limit 10;")
    schema, table = match.groups()
except:
    print("error input")
```



```python
re.findall("d(ds)*","askldjasaddsa") #-->['', 'ds']
re.findall("d[ds]*","askldjasaddsa") #-->['d', 'dds']
re.findall("d{ds}*","askldjasaddsa") #-->[]
re.findall("d{ds}","askldjasad{ds}a") #-->['d{ds}']
re.findall("ds?","askldjasad{dsssssss}a") #-->['d', 'd', 'ds']
```

非贪婪模式与贪婪模式：
```python
re.compile("/(\S*?)/").findall("//asd/sdsa/asda") #-->['', 'sdsa']
re.compile("/(\S*)/").findall("//asd/sdsa/asda")  #-->['/asd/sdsa']

#'*', '+'，和 '?' 修饰符都是贪婪的；它们在字符串进行尽可能多的匹配。
```

</br>

## _常用片段_


`[Jj]ava` 匹配 Java 和 java

`[1-3]` 或 `[123]` 匹配1、2或者3

--------------------

`<img.*src="(.*?)"` 从 html 的 img 标签中匹配出图片地址

--------------------
`(\w)(\w)\2\1` 可以匹配ABBA型数据（回溯引用）

-----------------
`http.*/$` 匹配以http开头,以/结尾的数据

------------------
`^\d{n}$`  匹配n位的数字

----------------
`^(?!(xx+)\1+$)x*` 匹配质数个x,不懂

--------------------
从`2002-1-2` `2020-01-02` `2020.01.02` `2020 01 02` `20200102` `2020/01/02`提取年月日的数据
`(\d{4})[\-/\s]?(\d{1,2})[\-/\s]?{\d{1,2}}`

-------------------------

`---title[\S\s]*?---` 非贪婪匹配，匹配文章的头信息。


</br>

## _特殊语法_

先行断言，后行断言，[参考资料](https://www.runoob.com/w3cnote/reg-lookahead-lookbehind.html)

比如 "a regular expression" 字符串，需要找到 regular 的 re，不能匹配 expression 的 re.

正向先行断言，如 `re(?=gular)`，限定了 re 右边的位置，这个位置之后是 gular, 但并不消耗 gular 这些字符

使用 `re(?=gular).` 匹配到了 reg

负向先行断言，如 `re(?!g)` 限定了 re 右边的位置不是字符 g

(?<=pattern) 正向后行断言

(?<!pattern) 负向后行断言

-------------

(?i) 实现对大小写不敏感的匹配






</br>


----------

参考资料：
- 官方文档：https://docs.python.org/zh-cn/3/library/re.html
- [正则表达式在线测试](http://c.runoob.com/front-end/854)
- 正则教程：https://codejiaonang.com/#/courses

