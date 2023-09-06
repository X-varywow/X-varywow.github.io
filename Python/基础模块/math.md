官方文档：https://docs.python.org/zh-cn/3/library/math.html

## （一）数论与表示函数
1. `math.ceil(x)`
返回 x 的上限，即大于或者等于 x 的最小整数。
2. `math.floor(x)`
返回 x 的向下取整，小于或等于 x 的最大整数。

3. `math.comb(n,k)`，**3.8新功能**
返回不重复且无顺序地从 n 项中选择 k 项的方式总数。
当 k <= n 时取值为 n! / (k! * (n - k)!)；当 k > n 时取值为零。
也称为二项式系数，因为它等价于表达式 (1 + x) ** n 的多项式展开中第 k 项的系数。

4. `math.fabs(x)`
返回 x 的绝对值。

5. `math.factorial(x)`
以一个整数返回 x 的阶乘。如果 x 不是整数或为负数时则将引发ValueError。

6. `math.gcd(*integers)`
求最大公约数，可以多个参数。

7. `math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)`
若 a 和 b 的值比较接近则返回 True，否则返回 False。
根据给定的绝对和相对容差确定两个值是否被认为是接近的。

8. `math.isinf(x)`
如果 x 是正或负无穷大，则返回 True ，否则返回 False 。

9. `math.modf(x)`
返回 x 的小数和整数部分。两个结果都带有 x 的符号并且是浮点数。

10. `math.perm(n, k=None)`，**3.8新功能**
排列不重复且无顺序地从 n 项中选择 k 项的方式总数。
当 k <= n 时取值为 n! / (n - k)!；当 k > n 时取值为零。
如果 k 未指定或为 None，则 k 默认值为 n 并且函数将返回 n!。

11. `math.prod(iterable, *, start=1)`，**3.8新功能**
计算输入的 iterable 中所有元素的积。 积的默认 start 值为 1。


## （二）幂函数与对数函数
12. `math.exp(x)`
返回 e 次 x 幂。

13. `math.log(x[, base ])`
使用一个参数时，返回 x 的自然对数（底为 e ）。

14. `math.sqrt(x)`
返回 x 的平方根。

15. `math.pow(x, y)`
返回 x 的 y 次幂。

## （三）三角函数与几何
16. `math.asin(x)`
以弧度为单位返回 x 的反正弦值。
 `math.acos(x)`
以弧度为单位返回 x 的反余弦值。
`math.atan(x)`
以弧度为单位返回 x 的反正切值。
`math.sin(x)`
返回 x 弧度的正弦值。
`math.cos(x)`
返回 x 弧度的余弦值。
`math.tan(x)`
返回 x 弧度的正切值。 结果范围在 -pi/2 到 pi/2 之间

17. `math.atan2(y, x)`
以弧度为单位返回 atan(y / x) ，结果在 `(-pi,pi]` 之间。

18. `math.dist(p, q)`，**3.8新功能**
返回 p 与 q 两点之间的欧几里得距离（两点间直线距离），`sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))`。以一个坐标序列（或可迭代对象）的形式给出。两个点必须具有相同的维度。

19.  `math.hypot(*coordinates)`
返回欧几里得范数，`sqrt(sum(x**2 for x in coordinates))`。这是从原点到坐标给定点的向量长度。

20. `math.degrees(x)`
将角度 x 从弧度转换为度数。
`math.radians(x)`
将角度 x 从度数转换为弧度。

## （四）常数

21. `math.pi`
数学常数 π = 3.141592...，精确到可用精度。

22. `math.e`
数学常数 e = 2.718281...，精确到可用精度。


---------------

**实践1**

```python
from math import *
ceil(3.9),floor(3.9)#-->4,3
gcd(3,-3)           #-->3
perm(5)，perm(5,2)  #-->120,20
log(100,10)         #-->2.0
dist((0,0),(2,2))   #-->2.8284271247461903
radians(180)        #-->3.141592653589793
```


**实践2**

```python
from math import tan, atan, pi

k2 = 1 
k2_1 = tan(atan(k2)-pi/6)   # 偏转 pi/6 的 k 值
k2_2 = tan(atan(k2)+pi/6)
```

----------------


本文基本包含math够用的函数。
未摘要的部分：
-  一些提高精度的函数
-  双曲函数
-  特殊函数，如伽马函数，统计等

发现3.8新功能好好啊。
