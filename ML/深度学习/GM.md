

> 主流的的生成模型：VAE, GAN, flow
- GAN
    - a generative model `G` that captures the data distribution
    - a discriminative model `D` that estimates the probability that a sample came from the training data rather than G
- VAE
- Flow


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230822004226.png" style="zoom:60%">


## GAN

Generative Adversarial Nets，生成对抗网络

原文地址：https://arxiv.org/abs/1406.2661

原理：生成器、判别器两者不断影响，目标是利用生成的数据替代真实的数据。最终使用生成器做生成就行了。

-----------

GAN 有着非常多的变种：DCGAN、WGAN 等


（DCGAN, pytorch 版）代码样例：

```python

```





-----------

参考资料：
- [万字详解什么是生成对抗网络GAN](https://bbs.huaweicloud.com/blogs/314916) ⭐️
- [GAN论文逐段精读【论文精读】](https://www.youtube.com/watch?v=g_0HtlrLiDo)
- [GAN-生成对抗网络原理及代码解析](https://www.bilibili.com/video/BV1ht411c79k/)


## VAE

变分自编码机(Variational AutoEncoder, VAE)，VAE作为可以和GAN比肩的生成模型，融合了贝叶斯方法和深度学习的优势，拥有优雅的数学基础和简单易懂的架构以及令人满意的性能，其能提取disentangled latent variable的特性也使得它比一般的生成模型具有更广泛的意义。

请参考：[ML/高级/vae](/ML/高级/vae)


## 扩散模型

属于无监督生成模型

如 Stable Diffusion (2022 年的深度学习文本到图像生成模型), [wiki](https://zh.wikipedia.org/zh-cn/Stable_Diffusion)。

[大一统视角理解扩散模型](https://zhuanlan.zhihu.com/p/558937247)

[Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)