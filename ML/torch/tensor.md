
https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

## 张量创建

```python
import tensor
import numpy as np

# 1. from list
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# 2. from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. from tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# 4. within
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

```

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(tensor.view)
```

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
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

## 张量运算

[course_02](https://www.bilibili.com/video/BV1wQ4y1q7Bm/)

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