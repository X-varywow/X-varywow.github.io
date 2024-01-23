
>总结：使用 cprofile 或 timeit 分析运行时间, 使用 memray 分析实时内存占用（将主体程序抽出来，分析 streamlit 程序有点不好）


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


## memory_profiler

```python
# pip install memory_profiler

import numpy as np
from memory_profiler import profile
import time

@profile
def demo():
    for i in range(5):
        a = np.random.rand(1000000)
        b = np.random.rand(1000000)
        
        a_ = a[a < b]
        b_ = b[a < b]

        time.sleep(2000)
    
    del a, b

    return a_, b_


if __name__ == '__main__':
    demo()
```


之后测试代码：

```bash
python tests/test_memory.py

# 使用 mprof 查看内存随时间的变化图
mprof run tests/test_memory.py

mprof plot
```

参考资料：https://zhuanlan.zhihu.com/p/121003986

STARS 4.1k , 更多：py-spy(11.4k)。 memray(11.8K)


## memray

```bash
pip install memray
memray run tests/test_memory.py

# 实时命令行界面
memray run --live my_script.py
# 在另一个端口打开，本窗口用于观测程序日志
memray run --live-remote ./tests/test_copy.py
memray3.9 live 52614

# 查看分析日志
memray tree
memray flamegraph tests/memray-test_memory.py.12585.bin
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

