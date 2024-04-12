
## _基本概念_

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






</br>


## _SD vs GAN_

GAN在训练时要同时训练生成器与判别器，所以其训练难度是比较大的。

生成器与判别器之间的权衡很难，也不能保证学习会收敛，通常会遇到梯度消失和模式崩溃等问题（当生成的样本没有多样性时）。

GAN 的训练收敛问题已通过扩散模型的发展得到解决。



</br>


## _SDXL_


（2023.08）

SDXL 1.0 也是所有开放式图像模型中参数量最多的模型之一，它建立在一个创新的新架构上，由一个 35 亿参数的基础模型和一个 66 亿参数的细化模型组成。

论文地址：https://arxiv.org/pdf/2307.01952.pdf

代码地址：https://github.com/Stability-AI/generative-models

官网：https://stablediffusionxl.com/

huggingface 在线：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0



[SDXL1.0评测](https://zhuanlan.zhihu.com/p/646879971)

[深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识](https://zhuanlan.zhihu.com/p/643420260)



</br>


## _Stability AI_

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
