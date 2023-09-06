

[2020 stylegan2](https://github.com/rosinality/stylegan2-pytorch) 2.4k [论文地址](https://arxiv.org/abs/1912.04958)

[2021 stylegan3](https://github.com/NVlabs/stylegan3) 5.6k


## 简介

整个网络结构还是保持了 PG-GAN （progressive growing GAN） 的结构

层级结构应对高清图像

style 来影响人脸的姿态、身份特征， noise 影响发丝、肤色等细节部分。

- 映射网络，生成中间隐变量来控制风格
- 合成网络

映射网络对隐藏空间 latent 解耦



## 应用1

>根据真实人物生成卡通人脸，再根据参考视频生成卡通动画

[参考](https://mp.weixin.qq.com/s?__biz=Mzg4NDQwNTI0OQ==&mid=2247522747&idx=1&sn=e7a75e897db57fc58a25f79679a849c3&source=41#wechat_redirect)

先使用卡通图像对 StyleGAN2 FFHQ人脸模型 fine-tune，再使用运动驱动模型：


[first-order-model](https://github.com/AliaksandrSiarohin/first-order-model/tree/master)

[huggingface demo](https://huggingface.co/spaces/abhishek/first-order-motion-model)


## 应用2

> 给定参考图像，并通过 StyleGAN v2 生成相似的人脸

Pixel2style2pixel(psp) 

https://github.com/eladrich/pixel2style2pixel


## 应用3

>使用 集成StyleGAN和采用卷积网络的图像转换框架，并移除图像生成的随机性、保持动画的连贯性，对视频中人物头像区域进行风格转化。

[VToonify](https://github.com/williamyang1991/VToonify)

## 应用4

animeganv2


----------------

参考资料：
- [StyleGAN 和 StyleGAN2 的深度理解](https://zhuanlan.zhihu.com/p/263554045)
- [Pixel2style2pixel(psp)簡介 — Encoding in Style a StyleGAN Encoder for Image-to-Image Translation](https://xiaosean5408.medium.com/pixel2style2pixel-psp-%E7%B0%A1%E4%BB%8B-encoding-in-style-a-stylegan-encoder-for-image-to-image-translation-7d2c9f2741d2)