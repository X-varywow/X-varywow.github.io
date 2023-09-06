
机器学习框架主要功能：
- 使用 GPU 计算
- 自动梯度微分
- 实现常见网络



```python
torch.autogard()
# 机器学习中的自动微分
```

```python
import torch

# 定义网络
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5,3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

# 计算梯度
loss.backward()
print(w.grad)
print(b.grad)
```

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)
# -> True

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
# -> False

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
# -> False
```

!>When we call backward for the second time with the same argument, the value of the gradient is different.


-------------

torch.mul、torch.mm、torch.bmm、torch.matmul的区别

- torch.mul，对位相乘，可以广播
- torch.mm，二位矩阵乘法
- torch.bmm，在 mm 基础上加了个 batch 计算，不能广播
- torch.matmul，能处理 batch ，广播


------------

参考资料：
- [torch 官方文档](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) ⭐️
- [Automatic Differentiation in ML: a survey](https://arxiv.org/pdf/1502.05767.pdf)
- https://www.cnblogs.com/rossiXYZ/p/15395742.html
- https://zhuanlan.zhihu.com/p/70302265
- https://openmlsys.github.io/chapter_frontend_and_ir/ad.html
- https://www.bilibili.com/video/BV1vL411u7bL/

