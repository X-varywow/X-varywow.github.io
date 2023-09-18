
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