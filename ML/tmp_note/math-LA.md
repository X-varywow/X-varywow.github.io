

</br>

## _点乘&叉乘_

> 点乘，表示向量 a 在 向量 b 上的投影

$$ a \cdot b = a_1b_1 + a_2b_2 + \cdots + a_nb_n$$

$$ a \cdot b = |a| |b| cos \theta$$


> 叉乘，又称外积、向量积，几何意义是以两个向量为边的平行四边形的面积

$$
A \times B = 
\begin{vmatrix}
a_1&a_2\\
b_1&b_2
\end{vmatrix}
= a_1b_2 - a_2b_1
$$


</br>

## _范数_

linalg: linear algebra norm (范数)

向量范数定义：
$$
\begin{aligned}
& \Vert \alpha \Vert_p = (|x_1|^p + |x_2|^p + \cdots + |x_n|^p)^\frac1p, p\ge1 \\ \\
& \Vert \alpha \Vert_1 = |x_1| + |x_2| + \cdots + |x_n| \\ 
& \Vert \alpha \Vert_2 = |s|
\end{aligned}
$$


应用1：

```python
# 判断 c 球是否遮挡在 a 球和 b 球中间

import numpy as np

R = 10

def _is_block(self, a, b, c):
    p1 = np.array([b[0] - a[0], b[1] - a[1]])
    p2 = np.array([c[0] - a[0], c[1] - a[1]])
    p3 = np.array([c[0] - b[0], c[1] - b[1]])
    if np.dot(p1, p2) >= 0 and np.dot(p1, p3) <= 0:    # 点乘
        d = self._calc_distance(c, a, b)
        return True if d < R else False
    else:
        return False
    
def _calc_distance(self, c, a, b):
    # if b is None:
    #     d = math.sqrt(math.pow(a[0] - self[0], 2) + math.pow(a[1] - self[1], 2))
    # else:
    p1 = np.array([a[0], a[1]])
    p2 = np.array([b[0], b[1]])
    p3 = np.array([c[0], c[1]])
    d = np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / np.linalg.norm(p2 - p1) if np.linalg.norm(p2 - p1) > 0 else 0.0 # 叉乘
    
    return d
```

## _几何_

```python
def target_point(dai, ball, d = R):

    delta_x = dai[0] - ball[0]
    delta_y = dai[1] - ball[0]

    d1 = dist(dai, ball)
    res = [0, 0]
    res[0] = dai[0] - delta_x * (d1+d)/d1
    res[1] = dai[1] - delta_y * (d1+d)/d1
    return res
```

## _矩阵_


[The-Art-of-Linear-Algebra](https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra) 没啥用，会还是会，不会还是不会

矩阵与图：https://thepalindrome.org/p/matrices-and-graphs

[调和矩阵，拉普拉斯矩阵](https://zh.wikipedia.org/zh-hans/%E8%B0%83%E5%92%8C%E7%9F%A9%E9%98%B5)

推荐：3B1W


- 方程组形式
- 几何意义（坐标系变化）