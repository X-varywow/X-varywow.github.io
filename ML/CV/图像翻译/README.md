


> 目标：对视频中的人物头像区域进行动漫化特效；

因为最终是制造高质量（效果好）的素材，所以换脸、风格转换、头部特效等，都会做一个尝试。


技术路线：
- 使用换脸技术
- 人脸识别，卡通化滤镜，动画特效，实时渲染
- 逐帧选取头部部分进行风格转化，然后使用视频剪辑技术拼接视频（VToonify）
- 选定一帧转化后作为参考，然后给定参考视频进行驱动来生成视频（first-order-motion）（Motion Representations for Articulated Animation）

------------

人像卡通风格渲染方案调研：
- https://github.com/keshav1990/HeadSwap
- [利用特定点位进行跟踪分析](https://www.bilibili.com/video/BV1ra4y1Y7B1/)，使用 AE
- [卡通风格转换](https://huggingface.co/spaces/PKUWilliamYang/VToonify)
- https://github.com/menyifang/DCT-Net
- [抖音超900万人在用的「卡通脸」特效技术揭秘](https://www.easemob.com/news/9937)，自研模型，也涉及 stylegan, pix2pix
- [PaddleGAN - Photo2cartoon](https://github.com/PaddlePaddle/PaddleGAN/blob/master/docs/zh_CN/tutorials/photo2cartoon.md)
- [AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch)