
scipy主要的模块

| 模块        | 功能           | 说明                                               | 常用函数示例                                   |
| ----------- | -------------- | -------------------------------------------------- | ---------------------------------------------- |
| cluster     | 聚类算法       | K-means、层次聚类、谱聚类等                        | `kmeans`, `hierarchy.linkage`                  |
| constants   | 物理数学常数   | 提供物理常数（光速、普朗克常数）和单位转换         | `c`, `pi`, `golden`                            |
| fftpack     | 快速傅里叶变换 | 离散傅里叶变换及其逆变换，频域分析                 | `fft`, `ifft`, `fftfreq`                       |
| integrate   | 积分与微分方程 | 数值积分、常微分方程（ODE）求解                    | `quad`, `dblquad`, `odeint`, `solve_ivp`       |
| interpolate | 插值           | 一维/多维插值、样条插值                            | `interp1d`, `griddata`, `UnivariateSpline`     |
| io          | 输入输出       | 读写 MATLAB .mat、WAV 音频、Matrix Market 等格式 | `loadmat`, `savemat`, `wavfile.read`           |
| linalg      | 线性代数       | 矩阵分解（LU/QR/SVD）、求解线性方程组、特征值      | `inv`, `det`, `eig`, `svd`, `solve`            |
| odr         | 正交距离回归   | 考虑自变量和因变量误差的回归分析                   | `ODR`, `Model`                                 |
| optimize    | 优化和求根     | 函数最小化、曲线拟合、方程求根                     | `minimize`, `curve_fit`, `fsolve`, `root`      |
| signal      | 信号处理       | 滤波器设计、卷积、频谱分析、峰值检测               | `butter`, `filtfilt`, `find_peaks`, `stft`     |
| sparse      | 稀疏矩阵       | 稀疏矩阵存储格式（CSR/CSC/COO）及运算              | `csr_matrix`, `csc_matrix`, `linalg.spsolve`   |
| spatial     | 空间算法       | KD树、Delaunay三角剖分、凸包、距离计算             | `KDTree`, `Delaunay`, `ConvexHull`, `distance` |
| special     | 特殊函数       | 贝塞尔函数、伽马函数、误差函数等数学特殊函数       | `jv`, `gamma`, `erf`, `comb`, `factorial`      |
| stats       | 统计           | 概率分布、假设检验、描述统计、相关性分析           | `norm`, `ttest_ind`, `pearsonr`, `kstest`      |
| ndimage     | N维图像处理    | 图像滤波、形态学操作、测量                         | `gaussian_filter`, `label`, `binary_dilation`  |


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


### method 参数详解

#### 无约束优化方法

| 方法        | 需要梯度 | 需要Hessian | 适用规模 | 收敛速度 |
| ----------- | -------- | ----------- | -------- | -------- |
| Nelder-Mead | ❌        | ❌           | 小规模   | 慢       |
| Powell      | ❌        | ❌           | 小规模   | 慢       |
| CG          | ✅        | ❌           | 大规模   | 中等     |
| BFGS        | ✅        | ❌           | 中规模   | 快       |
| L-BFGS-B    | ✅        | ❌           | 大规模   | 快       |
| Newton-CG   | ✅        | ✅           | 中规模   | 很快     |

#### 有约束优化方法

| 方法         | 边界约束 | 等式约束 | 不等式约束 |
| ------------ | -------- | -------- | ---------- |
| L-BFGS-B     | ✅        | ❌        | ❌          |
| SLSQP        | ✅        | ✅        | ✅          |
| trust-constr | ✅        | ✅        | ✅          |
| COBYLA       | ❌        | ✅        | ✅          |

---

**BFGS** (Broyden-Fletcher-Goldfarb-Shanno)
- 拟牛顿方法，通过迭代更新近似 Hessian 矩阵
- ✅ 优点：收敛速度快（超线性收敛），无需计算二阶导数
- ❌ 缺点：需要存储 n×n 矩阵，不适合高维问题；对初值敏感
- 🎯 适用：中等规模的光滑无约束问题

**L-BFGS-B** (Limited-memory BFGS with Bounds)
- BFGS 的有限内存版本，支持边界约束
- ✅ 优点：内存占用小，适合大规模问题；支持简单边界约束
- ❌ 缺点：不支持复杂约束（等式/不等式）
- 🎯 适用：大规模问题、变量有上下界限制

**SLSQP** (Sequential Least Squares Programming)
- 序列二次规划方法
- ✅ 优点：支持所有类型约束（边界、等式、不等式），收敛快
- ❌ 缺点：需要梯度信息，对非光滑问题效果差
- 🎯 适用：中小规模有约束优化问题（最常用的有约束方法）

**Nelder-Mead** (单纯形法)
- 基于单纯形搜索的直接方法
- ✅ 优点：无需梯度，对噪声函数鲁棒，实现简单
- ❌ 缺点：收敛慢，高维效果差，可能收敛到非最优点
- 🎯 适用：小规模问题、目标函数不可微或有噪声

**Powell**
- 共轭方向法，沿坐标方向交替搜索
- ✅ 优点：无需梯度，对某些问题收敛快
- ❌ 缺点：高维效率低，可能陷入局部最优
- 🎯 适用：小规模无约束问题

**CG** (Conjugate Gradient)
- 共轭梯度法
- ✅ 优点：内存需求低，适合大规模问题
- ❌ 缺点：对非二次函数收敛较慢，对条件数敏感
- 🎯 适用：大规模无约束问题

**Newton-CG**
- 截断牛顿法，结合牛顿法和共轭梯度
- ✅ 优点：收敛速度快（二次收敛），适合大规模问题
- ❌ 缺点：需要 Hessian 或 Hessian-vector 积
- 🎯 适用：大规模光滑问题、Hessian 稀疏时

**trust-constr**
- 信赖域约束优化
- ✅ 优点：处理约束能力强，收敛稳定，支持稀疏 Hessian
- ❌ 缺点：实现复杂，小问题可能不如 SLSQP 快
- 🎯 适用：大规模有约束问题、需要高精度解

**COBYLA** (Constrained Optimization BY Linear Approximation)
- 线性逼近约束优化
- ✅ 优点：无需梯度，支持约束
- ❌ 缺点：收敛慢，不支持边界约束（需转为不等式约束）
- 🎯 适用：目标函数或约束不可微

---

**选择建议：**
```
无约束 + 小规模 + 无梯度 → Nelder-Mead
无约束 + 中规模 + 有梯度 → BFGS
无约束 + 大规模 → L-BFGS-B 或 CG
有约束 + 通用场景 → SLSQP（首选）
有约束 + 大规模 → trust-constr
有约束 + 无梯度 → COBYLA
```