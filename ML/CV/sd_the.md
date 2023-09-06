
## 基本概念

- 组成
  - `Text Encoder`
    - CLIP pretrained transformer language model
  - `U-net model`
    - Image Information Creator
  - `VAE model`
    - Image Auto encoder-decoder
    - VAE 编码器，能将图像表示为低维特征，作为 U-Net 的输入
    - VAE 解码器，能将隐特征升维解码成完整图像
- use
  - diffusers
  - stable-diffusion webgui

-----------

核心概念：
- 噪声器
  - 通过剥离和产生噪声来形成图片


前向扩散过程涉及逐渐添加高斯噪声（即扩散过程），直到数据完全被噪声污染。

随后使用反向扩散方法对神经网络进行训练，以学习条件分布概率来反转噪声。


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230823220335.png" style="zoom:80%">




VAE 可以参考：[VAE 理解](ML/高级/vae)

-----------

finetune
- lora
- dreambooth
- hypernetwork
- textual inversion

-------------

一些提示：
- 素材最好超过50张。我是用了200张，每张图训练100到150步。素材选图很重要，尽量保持素材的质感一致，所以宁缺毋滥。比如数码照片和胶片照片质感的图一起训练，在还原胶片质感上，效果就会差很多



--------------

参考资料：
- [深入浅出完整解析Stable Diffusion核心基础知识](https://zhuanlan.zhihu.com/p/632809634)
- https://jalammar.github.io/illustrated-stable-diffusion/
- https://stable-diffusion-art.com/how-stable-diffusion-work/
- https://www.youtube.com/watch?v=dVjMiJsuR5o







## SD vs GAN

GAN在训练时要同时训练生成器与判别器，所以其训练难度是比较大的。

生成器与判别器之间的权衡很难，也不能保证学习会收敛，通常会遇到梯度消失和模式崩溃等问题（当生成的样本没有多样性时）。

GAN 的训练收敛问题已通过扩散模型的发展得到解决。




## Stability AI

[公司官网](https://stability.ai/)

sd 开源，生态友好；

盈利模式：
- 服务公司企业，提供定制模型和咨询服务
- 付费应用
- API

收购了 [clipdrop](https://clipdrop.co/tools)

使用 sagemaker 作为服务器，与 aws 并没有深层合作。

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230803232547.png">

------------

参考资料：
- https://new.qq.com/rain/a/20230602A0A5RF00
- [Stability AI 是如何赚钱的？ 稳定性 AI 商业模式分析](https://fourweekmba.com/zh-CN/%E7%A8%B3%E5%AE%9A%E6%80%A7ai%E6%98%AF%E6%80%8E%E4%B9%88%E8%B5%9A%E9%92%B1%E7%9A%84/)
