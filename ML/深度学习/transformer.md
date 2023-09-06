参考资料：
- 学校课程
- [论文原文：Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ⭐️ [中文版](https://blog.csdn.net/longxinchen_ml/article/details/86533005)
- [Transformer从零详细解读(可能是你见过最通俗易懂的讲解)](https://www.bilibili.com/video/BV1Di4y1c7Zm)
- [在线激情讲解transformer&Attention注意力机制](https://www.bilibili.com/video/BV1y44y1e7FW)
- [【Transformer模型】曼妙动画轻松学，形象比喻贼好记](https://www.bilibili.com/video/BV1MY41137AK)
- [手推transformer](https://www.bilibili.com/video/BV1UL411g7aX)
- https://blogs.nvidia.com/blog/2022/03/25/what-is-a-transformer-model/



很多经典的模型比如BERT、GPT-2都是基于Transformer的思想。

## 注意力机制

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220614001859.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220614001902.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220516160758.png">

$Qk^T$ 得到的会是两个矩阵每个行向量的点乘（用于反映相似度）



## 相关工作

[Attention机制竟有bug?](https://mp.weixin.qq.com/s/cSwWapqFhxu9zafzPUeVEw)


$$(softmax_1(x))_i = \frac{exp(x_i)}{1 + \sum_jexp(x_j)}$$


分母上加 1 将改变注意力单元，不再使用真实的权重概率向量，而是使用加起来小于 1 的权重。其动机是该网络可以学习提供高权重，这样调整后的 softmax 非常接近概率向量。同时有一个新的选项来提供 all-low 权重（它们提供 all-low 输出权重），这意味着它可以选择不对任何事情具有高置信度。




[xFormers](https://github.com/facebookresearch/xformers) - Toolbox to Accelerate Research on Transformers


