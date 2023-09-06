

## _优化器算法_

Optimizer，即优化损失函数的方法

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


</br>

在实际应用中，Adam 方法效果良好。

与其他自适应学习率算法相比，其收敛速度更快，学习效果更为有效，而且可以纠正其他优化技术中存在的问题，如学习率消失、收敛过慢或是高方差的参数更新导致损失函数波动较大等问题。

- 输入数据集比较稀疏时，SGD、动量等效果不好，应该使用自适应学习策略
- 若网络较为复杂，或想模型快速收敛，应该使用自适应学习策略


------------

参考资料：
- [从梯度下降到 Adam！一文看懂各种神经网络优化算法](https://www.cvmart.net/community/detail/5691)
- http://www.360doc.com/content/22/1031/20/32196507_1054056657.shtml