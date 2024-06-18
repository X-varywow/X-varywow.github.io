>数组二分查找算法

注意 sort() 不要带 reverse=True, 应该是个 bug 返回都是 0


### bisect
- `bisect.bisect_left(a,x,lo=0,hi=len(a))`
1. 在 a 中找到 x 合适的插入点以**维持有序**。
2. 如果 x 已经在 a 里存在，那么插入点会在已存在元素左边。
3. 如果 a 是`list`，则返回值可以是 `list.insert()` 的第一个参数。
4. 返回的`i`可以将数组 a 分成两部分。左侧是 `all(val < x for val in a[lo:i])` ，右侧是 `all(val >= x for val in a[i:hi])`

```python
import bisect
arr=[3,5,7,9,11,12,13]
x=8
bisect.bisect_left(arr,x)
# --> 3
```

- `bisect.bisect_right(a, x, lo=0, hi=len(a))`
- `bisect.bisect(a, x, lo=0, hi=len(a))`
类似于 `bisect_left()`，但是返回的插入点是 a 中已存在元素 x 的右侧。

### insort
- `bisect.insort_left(a, x, lo=0, hi=len(a))`
 1. 相当于 a.**insert**(bisect.bisect_left(a, x, lo, hi), x)
 2. 注意搜索是 O(log n) 的，插入却是 O(n) 的。

- `bisect.insort_right(a, x, lo=0, hi=len(a))`
- `bisect.insort(a, x, lo=0, hi=len(a))`
类似于 `insort_left()`，但是把 x 插入到 a 中已存在元素 x 的右侧。

### 例子

```python
def grade(score, breakpoints=[60,90], grades='FA'):
    i = bisect(breakpoints, score)
    return grades[i]

[grade(score) for score in [33,99,90]]

# output: ['F',  'A', 'A']
```

官方文档：https://docs.python.org/zh-cn/3.8/library/bisect.html?highlight=bisect#module-bisect
