官方文档：https://docs.python.org/zh-cn/3/library/functools.html

</br>

_functools_ 模块应用于高阶函数，即参数或（和）返回值为其他函数的函数。 通常来说，此模块的功能适用于所有可调用对象。

##### cache
简单轻量级未绑定函数缓存

-------------------

##### lru_cache

`@functools.lru_cache(maxsize=128, typed=False)`
一个为函数提供缓存功能的装饰器，缓存 maxsize 组传入参数，在下次以相同参数调用时直接返回上一次的结果。用以**节约高开销或I/O函数的调用时间**。

- 如果 maxsize 设为 None，LRU 特性将被禁用且缓存可无限增长。
- 如果 typed 设置为true，不同类型的函数参数将被分别缓存。例如， f(3) 和 f(3.0) 将被视为不同而分别缓存。

-----------------------

##### cmp_to_key

将比较函数作为key，示例代码：

```python
# 692. 前K个高频单词
from functools import cmp_to_key
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        def func(x,y):
            w1, c1 = x
            w2, c2 = y
            if c1>c2:
                return -1        #次数高的在前面
            elif c1==c2:         #次数相等的，字母序低的在前面
                return -1 if w1<w2 else 1
            else:
                return 1

        ans = Counter(words).most_common()
        ans.sort(key = cmp_to_key(lambda x,y:func(x,y)))
        return [i[0] for i in ans][:k]
```