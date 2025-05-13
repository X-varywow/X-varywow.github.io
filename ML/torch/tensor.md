

tensor 张量，是多维数组的核心数据结构, [torch 官方教程](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

## 张量创建


```python
import torch

# 从列表或NumPy数组创建
t1 = torch.tensor([1, 2, 3])          # 1D 张量（向量）
t2 = torch.tensor([[1, 2], [3, 4]])   # 2D 张量（矩阵）

# 特殊初始化
zeros = torch.zeros(2, 3)    # 全 0 张量
ones = torch.ones(2, 3)      # 全 1 张量
rand = torch.rand(2, 3)      # 均匀随机张量（0~1）
randn = torch.randn(2, 3)    # 标准正态分布随机张量

# 类似已有张量的形状
t3 = torch.rand_like(t2)     # 形状和 t2 相同
```

张量属性：

```python
x = torch.rand(2, 3)
print(x.dtype)    # 数据类型（默认 torch.float32）
print(x.shape)    # 形状（同 NumPy）
print(x.device)   # 存储设备（CPU/GPU）
```


```python
# 检测复数
torch.is_complex(input) -> (bool)
```

```python
torch.numel(a) # elem count
```

```python
torch.set_default_tensor_type(t)
```

```python
torch.arange(start, end, step=1)

# 比 arange 长一个单位
torch.range()
```

```python
# 转为 numpy.array
tensor.numpy()
```




## 张量运算



```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 逐元素运算
c = a + b       # 加法
d = a * b       # 乘法
e = torch.sin(a) # 数学函数

# 矩阵乘法
mat1 = torch.rand(2, 3)
mat2 = torch.rand(3, 2)
result = mat1 @ mat2   # 或 torch.matmul(mat1, mat2)

# 广播机制（Broadcasting）
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1], [2], [3]])
z = x + y  # 自动广播为 (3,3) + (3,3)
```





```python
# 分割
torch.chunk(b, chunks=2)
# 相对 chunk，可以指定份大小
torch.split()

torch.gather()

torch.reshape()

torch.tensor.scatter()

# 将维度为 1 的移除
torch.squeeze()

# 指定维度扩充 1 维
torch.unsqueeze(input, dim)

# 沿某个维度堆叠
torch.stack()

torch.transpose()

# 复制,铺砖式
torch.tile(a, [2,1])

# 返回切片
torch.unbind(input, dim=0)

# 条件
torch.where(x>0, x, y)
```

```python
torch.normal()

# [0, 1) 均匀分布
torch.rand()

torch.randint(low, high, size)

torch.randperm(4)
# -> [2,1,0,3]
# 用于构建样本
```

## 使用GPU


pytorch 不会自动将新张量放到 GPU 上，即使 CUDA 可用；**tensor 默认情况下创建在 CPU 上**；

.to(device) 是保持健壮性的通用做法


```python
import torch
import numpy as np

# 假设 CUDA 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从 NumPy 数组创建张量（默认在 CPU 上）
minibatch = [np.random.rand(3, 3) for _ in range(5)]
tensor = torch.FloatTensor(np.array([t[0] for t in minibatch])).unsqueeze(1)

print(tensor.device)  # 输出: cpu（即使 CUDA 已安装）

# 显式移动到 GPU
tensor = tensor.to(device)
print(tensor.device)  # 输出: cuda:0
```

gpu 加速：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)  # 移动到 GPU
y = y.to(device)
z = x + y         # GPU 加速计算
```

自动微分：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()       # 自动计算梯度
print(x.grad)      # dy/dx = 2x → 4.0
```




-----------

参考资料：
- [course_02](https://www.bilibili.com/video/BV1wQ4y1q7Bm/)
- gpt
