_introduction_
为高效循环而创建 **迭代器** 的函数。

官方文档：https://docs.python.org/zh-cn/3/library/itertools.html

### 排列组合迭代器

_product_
参数：`(*iterables, repeat=1)`
```python
product('AB', 'xy') #--> ('A', 'x')，('A', 'y')，('B', 'x')，('B', 'y')
product(range(2), repeat=3) #--> 2*2*2，8个元组

# 笛卡儿积
```


_permutations_
排列，参数：`(iterable, r=None)`

```python
permutations('ABCD', 2)  # --> AB AC AD BA BC BD CA CB CD DA DB DC
permutations(range(3))   # --> 012 021 102 120 201 210
```


_combinations_
组合，参数：`(iterable, r)`

```python
combinations('ABCD', 2)   # --> AB AC AD BC BD CD
combinations(range(4), 3) # --> 012 013 023 123
```

*combinations_with_replacement*
允许重复的组合，参数：`(iterable, r)`

```python
combinations_with_replacement('ABC', 2) #--> AA AB AC BB BC CC

#允许元素重复出现
```



### 无穷迭代器

- `count()`
- `cycle()`
- `repeat()`

```python
from itertools import *

count(10)   # 10,11,12...

cycle('ABC') # A, B, C, A, B...

repeat(x [,n])
```

### 其它迭代器

_accumulate_

参数：`(iterable[, func, *, initial=None])`
```python
accumulate([1,2,3,4,5])               #--> 1 3 6 10 15
accumulate([1,2,3,4,5], initial=100)  #--> 100 101 103 106 110 115
accumulate([1,2,3,4,5], operator.mul) #--> 1 2 6 24 120
```

_compress_	

```python
compress('ABCDEF', [1,0,1,0,1,1])  #--> A C E F
```

*zip_longeset*

参数：`itertools.zip_longest(*iterables, fillvalue=None)`

创建一个迭代器，从每个可迭代对象中收集元素。如果可迭代对象的长度未对齐，将根据 fillvalue 填充缺失值。迭代持续到耗光最长的可迭代对象。