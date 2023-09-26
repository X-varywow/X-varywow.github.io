
启发式算法 Heuristic Algorithm

一种基于经验的算法，用来优化整个探索的过程。

- 模拟退火算法
- 遗传算法（不断删除指标低的）
- 蚁群算法
- 差分进化算法
- 鱼群算法
- 免疫优化算法
- 粒子群优化算法(Particle Swarm Optimization, PSO) 




```python
!pip install scikit-opt
```

</br>

_差分进化_

STEP1：定义约束优化问题
```python
'''
min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5
'''


def obj_func(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2


constraint_eq = [
    lambda x: 1 - x[1] - x[2]
]

constraint_ueq = [
    lambda x: 1 - x[0] * x[1],
    lambda x: x[0] * x[1] - 5
]
```

STEP2：差分进化
```python
from sko.DE import DE

de = DE(func=obj_func, n_dim=3, size_pop=50, max_iter=800, lb=[0, 0, 0], ub=[5, 5, 5],
        constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
```






----------------

参考资料：
- [七种启发式算法](https://zhuanlan.zhihu.com/p/371637604)
- [scikit-opt 官方文档](https://scikit-opt.github.io/scikit-opt/#/zh/README)
- [粒子群优化算法(Particle Swarm Optimization, PSO)的详细解读](https://zhuanlan.zhihu.com/p/346355572)