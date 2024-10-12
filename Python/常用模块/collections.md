[官方文档链接](https://docs.python.org/zh-cn/3/library/collections.html)

## Counter

- 一个 `Counter` 是一个 `dict` 的子类，用于计数可哈希对象。
- 元素从一个 `iterable` 被计数或从其他的 `mapping (or counter)`初始化.
- 设置一个计数为0不会从计数器中移去一个元素。使用 `del` 来删除它

##### 初始化:

```python
from collections import *
c = Counter()                           # a new, empty counter
c = Counter('gallahad')                 # a new counter from an iterable
c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
c = Counter(cats=4, dogs=8)             # a new counter from keyword args
```

##### elements()

```python
#elements()返回一个迭代器
c = Counter(a=4, b=2, c=0, d=-2)
sorted(c.elements())   #-->['a', 'a', 'a', 'a', 'b', 'b']
```

##### most_common()

```python
#相等个数的元素按首次出现的顺序排序：
Counter('abracadabra').most_common(3)   #-->[('a', 5), ('b', 2), ('r', 2)]
```

##### subtract()

```python
c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
c.subtract(d)
c            #-->Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})
```

##### 一些用法：

```python
sum(c.values())                 # total of all counts
c.clear()                       # reset all counts
list(c)                         # list unique elements
set(c)                          # convert to a set
dict(c)                         # convert to a regular dictionary
c.items()                       # convert to a list of (elem, cnt) pairs
Counter(dict(list_of_pairs))    # convert from a list of (elem, cnt) pairs
c.most_common()[:-n-1:-1]       # n least common elements
+c                              # remove zero and negative counts
```

```python
import re
words = re.findall(r'\w+', open('hamlet.txt').read().lower())
Counter(words).most_common(10)
'''-->[('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631),
  ('you', 554),  ('a', 546), ('my', 514), ('hamlet', 471), ('in', 451)]'''
```

## deque
类似`list`的容器，实现了在两端快速`append`和`pop`
- `append(x)`
- `appendleft(x)`
- `clear()`
- `copy()`  浅拷贝
- `count(x)`
- `extend()`
- `insert(i,x)`
- `pop()`
- `popleft()`
- `reverse()`

## namedtuple()
创建命名元组子类的工厂函数

## defaultdict


字典的子类，提供了一个工厂函数，为字典查询提供一个默认值。

常见用法：
```python
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)
sorted(d.items())  #-->[('blue', [2, 4]), ('red', [1]), ('yellow', [1, 3])]
```

借助 lambda 传入自定义结构：
```python
data = defaultdict(lambda : [0]*8)
```



## OrderedDict

返回一个 dict 子类的实例，它具有专门用于重新排列字典顺序的方法。

>利用该数据类型，可以简易实现LRU缓存（Least Recently Used）

```python
from collections import OrderedDict
class LRUCache:

    def __init__(self, capacity: int):
        self.d = OrderedDict()
        self.n = capacity

    def get(self, key: int) -> int:
        if key in self.d:
            self.d.move_to_end(key)
            return self.d.get(key)
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            del self.d[key]
            self.d[key] = value
        else:
            if len(self.d) == self.n:
                self.d.popitem(last = False)
            self.d[key] = value
```
