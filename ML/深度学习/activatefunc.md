

思考车厢之间的耦合结构，这种结构不仅仅可以连接车厢，并且可使车厢相对旋转，使列车具有转向功能。

感知机，本意是模拟神经元的刺激传导，给输入乘以权重后设定阈值进行判定。

**激活函数**，就是其中的阈值判定部分。



## _Sigmoid_


$$ f(x) = \frac{1}{1+exp(-x)}$$



</br>

## _Tanh/双曲正切_

$$ f(x) = tanh(x) = \frac{2}{1+exp(-2x) - 1}$$


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231102215817.png">

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

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231102220126.png">




</br>

## _Softmax_

Softmax函数，它将一个数值向量，转化为一个和为 1 的概率分布向量。


$$Probability = \frac{exp(z_j)}{\sum_{j=1}^Kexp(z_j)}$$

-------

代码实现如下：

```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

x = np.array([1, 2, 3, 4, 5])  # 输入向量
y_softmax = softmax(x)
```


它是一个<u>多分类函数</u>，无法直接绘制曲线，是根据输入向量得到的一个概率分布向量。

按照上述 [1, 2, 3, 4, 5] 输入，结果如下：

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231102221531.png">



-----------

为什么使用 e^x 而不是 x ？ 为了反向传播中的求导

为什么要减去 max(x) ？为了防止（指数结果）数值溢出，且不会改变原本性质

</br>

## _Maxout_

Maxout 激活函数并不是一个固定的函数，不像Sigmod、Relu、Tanh等函数，是一个固定的函数方程.它是一个可学习的激活函数，W 参数是学习变化的。


只需2个 maxout 节点就可以拟合任意的凸函数

优点：拟合能力非常强，可以拟合任意的凸函数。Maxout具有ReLU的所有优点，线性、不饱和性。同时没有ReLU的一些缺点。如：神经元的死亡。

缺点：参数量较大






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
- [深度学习领域最常用的10个激活函数](https://www.jiqizhixin.com/articles/2021-02-24-7)
- chatgpt