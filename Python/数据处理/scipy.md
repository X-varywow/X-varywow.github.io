
scipy主要的模块

cluster 聚类算法
constants 物理数学常数
fftpack 快速傅里叶变换
integrate 积分和常微分方程求解
interpolate 插值
io 输入输出
linalg 线性代数
odr 正交距离回归
optimize 优化和求根
signal 信号处理
sparse 稀疏矩阵
spatial 空间数据结构和算法
special 特殊方程
stats 统计分布和函数
weave C/C++ 积分


```python
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt



# 最小二乘拟合
opt.leastsq(func, x0)
```

[机器量化分析（九）—— Scipy库的使用与优化问题](https://zhuanlan.zhihu.com/p/349321907)






</br>

## _lsq\_linear_

求解具有变量界限的最小二乘问题

minimize 0.5 * ||A x - b||**2
subject to lb <= x <= ub

demo: 3x1 + 4x2 + 5x3 = 4.8， x1+x2+x3 = 1

```python
from scipy.optimize import lsq_linear

x_arr = [3,4,5]
y = 4.8

tt = [1] * len(x_arr)
A = np.array([x_arr, tt])
b = np.array([y, 1])

bounds = (0.01, 1)

# 使用lsq_linear函数求解带有范围约束的最小二乘问题
res = lsq_linear(A, b, bounds=bounds)
p = [round(prob, 4) for prob in res.x]

res = dict(zip(x_arr, p))
print(res)
# {3: 0.0413, 4: 0.1174, 5: 0.8413}
```