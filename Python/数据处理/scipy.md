
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

求解线性最小二乘问题

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

</br>

## _optimize.minimize_


demo:

```python
n = len(hml)

# cons1: cumulate p is 1

A = np.array([
    [WEIGHT_P for _ in range(n)],
    [i*WEIGHT_RR for i in arr1],
    [i*WEIGHT_MC for i in arr2],
    [i*WEIGHT_HL for i in arr3]
])
b = np.array([WEIGHT_P, WEIGHT_RR, WEIGHT_MC, 0])
# res = lsq_linear(A, b, bounds=(0,1))
# best_x = res.x

def regularization(x):
    return np.sum((np.maximum(0.02 - x, 0))**2)

def objective(x):
    return np.linalg.norm(A @ x - b) + WEIGHT_RG*regularization(x)

res = minimize(objective, x0=np.array([1/n for _ in range(n)]), method='SLSQP', bounds = [(0, 1) for _ in range(n)])
best_x = res.x
```


method:
- BFGS （拟牛顿方法，用于求解无约束优化问题，需要梯度信息）
  - Broyden-Fletcher-Goldfarb-Shanno
- SLSQP（一种用于求解有约束优化问题的算法，需要梯度信息）
  - Sequential Least Squares Programming
- Nelder-Mead （基于单纯形搜索的直接搜索方法，不需要计算目标函数的梯度）