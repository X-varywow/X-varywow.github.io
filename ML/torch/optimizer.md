

## _优化器算法_

Optimizer，即优化损失函数的方法

优化器算法，通过计算损失函数的梯度来确定参数的更新方向和步长，以逐步调整模型的参数来优化模型。

</br>


常见优化器：
- Batch Gradient Descent, `BGD`
- Stochastic Gradient Descent, `SGD`
- Mini-Batch Gradient Descent, `MBGD`
- Momentum
  - 动量，通过优化相关方向的训练和弱化无关方向的振荡，来加速SGD训练。
- Nesterov Accelerated Gradient
- Adagrad
- AdaDelta
- `Adam`
  - Adaptive Moment Estimation，能计算每个参数的自适应学习率
  - 2015 年提出，融合了 Momentum 和 AdaGrad 的方法
- `AdamW`
  - 相比于 Adam, 主要的改进在权重衰减的处理上
  - 传统的 Adam 算法在更新参数时会对权重进行衰减，但是这种衰减方式不符合 L2 正则化的定义，可能会导致模型收敛不理想。而 AdamW 算法则将权重衰减应用到了梯度更新的步骤中，使得模型的收敛性更好


</br>

Adam 方法更加实用，

与其他自适应学习率算法相比，其收敛速度更快，学习效果更为有效，而且可以纠正其他优化技术中存在的问题，如学习率消失、收敛过慢或是高方差的参数更新导致损失函数波动较大等问题。

- 输入数据集比较稀疏时，SGD、动量等效果不好，应该使用自适应学习策略
- 若网络较为复杂，或想模型快速收敛，应该使用自适应学习策略


-----------

- optim.Adagrad
- optim.Adam
- optim.AdamW
- optim.ASGD
- optim.SGD






------------

参考资料：
- [torch-optim](https://pytorch.org/docs/stable/optim.html)
- [从梯度下降到 Adam！一文看懂各种神经网络优化算法](https://www.cvmart.net/community/detail/5691)
- http://www.360doc.com/content/22/1031/20/32196507_1054056657.shtml
- chatgpt