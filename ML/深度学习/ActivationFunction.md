

思考车厢之间的耦合结构，这种结构不仅仅可以连接车厢，并且可使车厢相对旋转，使列车具有转向功能。

感知机，本意是模拟神经元的刺激传导，给输入乘以权重后设定阈值进行判定。

**激活函数**，就是其中的阈值判定部分。



## _Sigmoid_


$$ h(x) = \frac{1}{1+exp(-x)}$$



</br>

## _Tanh/双曲正切激活函数_


</br>

## _ReLU_


$$
h(x) = \begin{cases}
x \quad (x>0)\\
0 \quad (x\le 0)
\end{cases}$$

为了缓解梯度消失的问题，梯度非 0 即 1；这意味着梯度连乘不会收敛到 0，

若为0则梯度停止前向传播，为网络引入了稀疏性。

但是，梯度为 0 会导致神经元“死亡”（无法参与学习），于是引入了 Leaky ReLU:

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231026231755.png">





</br>

## _SeLU_


</br>

## _Softmax_

[关于Softmax函数](https://zhuanlan.zhihu.com/p/168562182)，它将一个数值向量归一化为一个概率分布向量。

softmax 输出值总和为 1


</br>

## _Maxout_



</br>

## _other_


阶跃函数是激活函数的区别：
- 阶跃函数是激活函数中的一种，在神经网络早期被广泛使用。
- 后续又引入了各种非线性性质的函数（能够进行非线性变换），如 sigmoid, relu, tanh 等，增加网络的表达能力。
- 神经网络中常使用平滑变化的 sigmoid 函数

**阶跃函数容易抹消参数的微小变化**


------------------

参考资料：
- [深度学习笔记：如何理解激活函数？（附常用激活函数）](https://zhuanlan.zhihu.com/p/364620596)
- [更多的激活函数注解](https://cloud.tencent.com/developer/article/1800954)
- https://www.jiqizhixin.com/articles/2021-02-24-7