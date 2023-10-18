

mpmath is a Python library **for real and complex floating-point arithmetic** with arbitrary precision


官方网址：https://mpmath.org/

```python
from mpmath import mp

mp.dps = 20
print(mp.quad(lambda x: mp.exp(-x**2), [-mp.inf, mp.inf]) ** 2)

# -> 3.1415926535897932385
```