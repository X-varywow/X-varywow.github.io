

## _数据变换_

对数据的变换操作，是 **使其尽可能符合我们的假设**


### _（1）对数变换_

[在统计学中为什么要对变量取对数？](https://www.zhihu.com/question/22012482)

- 单调增函数，不会改变数据的相对关系，但 **压缩了变量的尺度**
- 取对数后，乘法运算变为加法运算，线性关系更方便做参数估计


取对数可以将大于中位数的值按一定比例缩小，从而形成正态分布的数据，对于计量模型，解决方差问题都有很大帮助。




### _（2）BOX-COX变换_

参考：[数据不够正态，Box-Cox 来转换](https://zhuanlan.zhihu.com/p/654738505)

用于连续的响应变量不满足正态分布的情况，可以一定程度上减小不可观测的误差和预测变量的相关性。

明显地 **改善数据的正态性、对称性和方差稳定性，对偏度的矫正**

通过最大似然估计 $\lambda$ 转换参数，使其尽可能接近正态分布


$$
y(\lambda) = \begin{cases}
\begin{aligned}
\frac{y^{\lambda}-1}{\lambda},   \quad\quad &if \ \lambda \neq 0 \\
ln(y),        \quad\quad&if\  \lambda = 0
\end{aligned}
\end{cases}
$$


</br>

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

# 数据
crime_rates = [23, 151, 66, 46, 8, 8, 3, 101, 46, 62, 1, 175, 89, 12, 10, 10, 18, 37, 28, 17]

# 绘制频率分布直方图和概率密度曲线
plt.figure(figsize=(10,6))

# kde = True 绘制核密度曲线
# bins = 10 数据分成10个区间绘制直方图
sns.histplot(crime_rates, kde=True, bins=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)



plt.title('Frequency Distribution of Crime Rates')
plt.xlabel('Crime Rate (per 10,000)')
plt.ylabel('Frequency')
plt.show()

```

```python
# 检验正态性

shapiro_test = shapiro(crime_rates)
shapiro_test


# w 值 0.8087
# P 值 0.0012 < 0.05 , 拒绝原假设（数据正态）
```


```python
from scipy.stats import boxcox

# 应用Box-Cox转换
transformed_data, best_lambda = boxcox(crime_rates)

best_lambda, transformed_data

# len(transformed_data) == len(crime_rates)
```

这直接把原始数据给转换了，效果会好吗？保留了原始数据的哪些信息？



大小关系没有变化



试一下，就这么个矫正非正态的工具

seaborn scipy.stats






</br>

## _偏态分布_

可通过峰度和偏度的计算，衡量偏态的程度。


```python
import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
a = 4
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')

x = np.linspace(skewnorm.ppf(0.01, a),
                skewnorm.ppf(0.99, a), 100)
ax.plot(x, skewnorm.pdf(x, a),
       'r-', lw=5, alpha=0.6, label='skewnorm pdf')
rv = skewnorm(a)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = skewnorm.ppf([0.001, 0.5, 0.999], a)
np.allclose([0.001, 0.5, 0.999], skewnorm.cdf(vals, a))
r = skewnorm.rvs(a, size=1000)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])
ax.legend(loc='best', frameon=False)
plt.show()
```


--------------

参考资料：
- [scipy.stats.skewnorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html)
- [讲讲偏态分布](https://zhuanlan.zhihu.com/p/367865378)
