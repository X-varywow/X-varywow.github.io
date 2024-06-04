

## _cProfile_

使用 cProfile 分析性能：

```python
python -m cProfile -s tottime your_program.py

# 重定向 -> test.txt
python -m cProfile -s tottime ZZZ_log.py > test.txt
```

- tottime，指的是函数本身的运行时间，扣除了子函数的运行时间
- cumtime，指的是函数的累计运行时间，包含了子函数的运行时间

更多性能分析模块： line_profiler, pyflame

-----------


代码中使用：

```python
import cProfile

cProfile.run('your_func()')
```

```python
import cProfile
import pstats

def my_function():
    # 你的代码逻辑
    pass

# 运行性能分析
profiler = cProfile.Profile()
profiler.enable()
my_function()
profiler.disable()

# 使用 pstats 来读取和分析数据
stats = pstats.Stats(profiler)

# 按照总时间（tottime）排序并打印前30个函数
stats.sort_stats('tottime').print_stats(30)
```







## _测试时间_

```python
import time

start = time.time()
foo()
print(time.time() - start)
```

```python
%%time
# 运行这个 cell 所需时间
```

```python
%time  1+1  #当前行的代码运行一次所花费的时间

%timeit    #比较常用，测试时间

# 使用 %% 可以换行，测的是整个 jupyter cell
```



demo1: numpy 加速计算并测试

```python
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

arr = np.random.randint(1, 100, size = 100000)

def way1():
    tot = 0
    cnt = 0
    for num in arr:
        if num<70:
            tot += num
            cnt += 1
    return tot/cnt

def way2():
    res = 0
    res = arr[arr<70].mean()
    return res

%timeit way1()

%timeit way2()

# -> 14.5 ms ± 36.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# -> 717 µs ± 7.15 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 性能是其 20 倍
```

## 使用 jit 优化性能

python 是解释型语言，动态数据类型，也导致速度较慢。

JIT(just-in-time) 对于（数值运算类）（循环类）能大幅提速，list和特殊对象 不合适



-----------

demo1:

```python
from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

print(go_fast(x))
```


-----


demo2：

```python
from numba import jit

def way3(c, a, b):
    
    x1, y1 = b[0]-a[0], b[1]-a[1]
    x2, y2 = c[0]-a[0], c[1]-a[1]
    
    d = abs((x1*y2 - y1*x2)) / sqrt(x1*x1 + y1*y1)

    return d

@jit()
def way4(cx, cy, ax, ay, bx, by):
    
    x1, y1 = bx-ax, by-ay
    x2, y2 = cx-ax, cy-ay
    
    d = abs((x1*y2 - y1*x2)) / sqrt(x1*x1 + y1*y1)

    return d

%timeit way3([3.3, 6.6],[9.9,19.8],[4.4,8.8]) 
# -> 700 ns ± 2.32 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

%timeit way4(3.3, 6.6, 9.9,19.8,4.4,8.8)
# -> 291 ns ± 2.51 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

> `nopython=True` 时会脱离 python 解释器，导致调试时函数内部断点无效。

## line_profiler

自动生成行级代码的性能分析， [官方文档](https://kernprof.readthedocs.io/en/latest/)


```python
# demo_primes.py
from line_profiler import profile

@profile
def func():
    pass

if __name__ == '__main__':
    func()
```


```bash
# 会生成 3 个文件
LINE_PROFILE=1 python demo_primes.py

# bash 中显示
python -m kernprof -lvr demo_primes.py
```



## other 

pypy 即时编译器替代 CPython 解释器，pypy 中有 JIT


numba  还有的功能：
- automatic parallelization
- fast-math


一些优化点：
- 不使用 numpy 方法，使用原生方法
- 不使用 listcomp，减少中间变量
- 函数参数不使用 列表, 少的情况直接取出来


-----------

参考资料：
- [使用 cProfile 和火焰图调优 Python 程序性能](https://zhuanlan.zhihu.com/p/53760922)
- https://jiffyclub.github.io/snakeviz/
- [cprofile 官方文档](https://docs.python.org/zh-cn/3/library/profile.html)
- [Python 程序运行时间测试](https://juejin.cn/post/7028597865547038728)
- [timeit 官方文档](https://docs.python.org/zh-cn/3/library/timeit.html)
- [numba 官方文档](https://numba.pydata.org/)
- [(超级详细）jit的介绍和用法（python加速）](https://blog.csdn.net/qq_43391414/article/details/123248978)
- [wiki - Numba](https://zh.wikipedia.org/wiki/Numba)

