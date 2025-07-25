



_（1）全连接层_

也称为FC或密集层（dense layer），是一组从前一层 **所有神经元** 那里接收输入的神经元的集合



</br>

_（2）激活函数_

参考左侧：激活函数




</br>

_（3）dropout_

一种延缓过拟合的方法；。

dropout层不包含任何神经元，不做任何计算。相反，它 **使前一层的部分神经元断开连接**（20%）。

这种层仅仅在训练时有效，当我们使用网络来预测时，dropout层是没有影响的。

[为什么训大模型都不用dropout](https://mp.weixin.qq.com/s/2GOHIsBBGHBA9yEPk6Ztrw)

待验证：在以前小数据集模型时，有些用；但存在的问题（训练推理行为不一致，增加训练资源）后续比较严重。

</br>

_（4）batchnorm_


批归一，与（3）一样也是一种正则化技术



</br>

_（5）卷积层_

扩展到矩阵的点积

如卷积核：
$kernel = \begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}$ 会提取图像的 垂直边缘信息

demo:
`nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)` 

表示用 32 个 3×3 的卷积核 对 单通道输入 进行扫描，步长1，填充1，使输出尺寸和输入相同，并提取 32 种不同的特征。（卷积核的权重会在训练过程中被不断更新, nn.Conv2d.weight 查看）

$$输出尺寸 = [\frac{输入尺寸 + 2*padding - kernel\_size}{stride}]+1$$


</br>

_（6）池化层_

主要是为了简化矩阵的结构（简化特征），eg. 矩阵中每4个元素抽成1个

- 最大池化
- 平均池化

[CNN基础知识——池化（pooling）](https://zhuanlan.zhihu.com/p/78760534)


</br>

_（7）循环层_


------------------

参考资料：
- [万字详解什么是生成对抗网络GAN](https://bbs.huaweicloud.com/blogs/314916)
- 地毯书，第20章：深度学习