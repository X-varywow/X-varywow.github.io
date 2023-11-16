

mpmath is a Python library **for real and complex floating-point arithmetic** with arbitrary precision



```python
from mpmath import mp

# 设置精度
mp.dps = 20
print(mp.quad(lambda x: mp.exp(-x**2), [-mp.inf, mp.inf]) ** 2)

# -> 3.1415926535897932385


mp.sin(1)
```

number types: 
- mpf, real float
- mpc, complex float
- matrix

------------

其它具有类似功能的模块：decimal, 

```python
import decimal

a = decimal.Decimal('0.10')
b = decimal.Decimal('0.20')

c = a + b
print(c)        # 0.30
```





------------

参考资料：
- [mpmath basics](https://mpmath.org/doc/current/basics.html)