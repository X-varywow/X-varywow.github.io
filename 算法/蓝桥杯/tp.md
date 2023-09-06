
> 本文作为熟练 python ，并从题目中见过大多模块，的总结；

## 常见函数

#### 1. 计时装饰器

```python
import functools
import time

# functools.wraps 旨在消除装饰器对原函数造成的影响
def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        time_cost = time.time() - start_time
        print(func.__name__ + " time_cost: {} s".format(time_cost))
        return res
    return clocked
```

#### 2. 排序传入函数
```python
# leetcode937. 重新排列日志文件
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def trans(log: str) -> tuple:
            a, b = log.split(' ', 1)
            return (0, b, a) if b[0].isalpha() else (1,)

        logs.sort(key=trans)  # sort 是稳定排序
        return logs
```

#### 3. 并查集

```python
f = {}
def find(x):
    f.setdefault(x, x)
    while x != f[x]:
        f[x] = f[f[x]]
        x = f[x]
    return x
def union(x, y):
    f[find(x)] = find(y)
```

#### 4. 回溯

```python
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```

----------------------


[::-1]

bin() hex() eval()

zip() reduce() * map()

enumerate()

字符串的各种函数：

| 函数                               | 说明                                                                                                                    |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| isalnum()                          | 如果字符串至少有一个字符并且所有字符都是字母或数字则返 回 True，否则返回 False                                          |
| isalpha()                          | 如果字符串至少有一个字符并且所有字符都是字母或中文字则返回 True, 否则返回 False                                         |
| isdigit()                          | 如果字符串只包含数字则返回 True 否则返回 False..                                                                        |
| islower()                          | 如果字符串中包含至少一个区分大小写的字符，并且所有这些(区分大小写的)字符都是小写，则返回 True，否则返回 False           |
| isupper()                          | 如果字符串中包含至少一个区分大小写的字符，并且所有这些(区分大小写的)字符都是大写，则返回 True，否则返回 False           |
| lower()                            | 转换字符串中所有大写字符为小写                                                                                          |
| upper()                            | 转换字符串中的小写字母为大写                                                                                            |
| strip()                            | 删除字符串开头和结尾的空格或指定字符                                                                                    |
| find(str, beg=0, end=len(string))  | 检测 str 是否包含在字符串中，如果指定范围 beg 和 end ，则检查是否包含在指定范围内，如果包含返回开始的索引值，否则返回-1 |
| index(str, beg=0, end=len(string)) | 跟find()方法一样，只不过如果str不在字符串中会报一个异常                                                                 |
| startswith()                       | 检查字符串是否以 obj 开始                                                                                               |
| endswith()                         | 检查字符串是否以 obj 结束                                                                                               |

## 常见模块

#### 1. collections

deque 双端队列，可设置参数 maxlen

Counter 计数器

defaultdict

OrderedDict

#### 2. heapq

最小堆，默认 pop 出来的是最小的元素

```python
heapify(x)
#将list x 转换成堆，原地，线性时间内。

heappush(heap, item)
#将 item 的值加入 heap 中，保持堆的不变性。

heappop(heap)
#弹出并返回 heap 的最小的元素，保持堆的不变性。

heappushpop(heap, item)
#将 item 放入堆中，然后弹出并返回 heap 的最小元素。
```


#### 3. itertools


```python
# 排列
permutations('ABCD', 2)  # --> AB AC AD BA BC BD CA CB CD DA DB DC
permutations(range(3))   # --> 012 021 102 120 201 210

# 组合
combinations('ABCD', 2)   # --> AB AC AD BC BD CD
combinations(range(4), 3) # --> 012 013 023 123

# 其它
itertools.zip_longest(*iterables, fillvalue=None)
```

#### 4. functools

```python
from functools import lru_cache

@lru_cache(None)
def main():
    pass
```

#### 5. math

```python
from math import *

ceil()
floor()
fabs()       # 绝对值
factorial(x) # 阶乘

gcd(*integers) #求最大公约数，可以多个参数。
lcm(*integers) #求最小公倍数，可以多个参数。

comb(n, k) # 返回组合数
perm(n, k) # 返回排列数

dist(p, q) # 返回两点间直线距离
```


#### 6. datetime

```python
from datetime import datetime, timedelta

start = datetime(2001,1,1)
end = datetime(2021,12,31)
delta = timedelta(1)

start += delta
start.year
start.month
start.day
```

[Python datetime模块详解、示例](https://blog.csdn.net/cmzsteven/article/details/64906245)


#### 7. bisect

数组二分查找算法

```python
import bisect
arr=[3,5,7,9,11,12,13]
x=8
bisect.bisect_left(arr,x)
# --> 3
```



#### 8. other


- queue
  - PriorityQueue


