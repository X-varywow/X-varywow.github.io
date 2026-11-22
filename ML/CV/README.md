
欢迎来到 计算机视觉

主要任务：图像分类、目标检测、图像分割、图像合成

2026.11，发展太快了(seeddance 2.5等)，新一代多模态生成模型已经能完成过去的很多专门链路，如抠图，数字人，换脸，看下仓库好多都是已归档状态。。。

现在生图模型基本是多模态模型

---------

comfyUI 生图链路上的模型架构知识：


技术演进路线：
- 2022：Latent Diffusion + U-Net
  - Stable Diffusion 1.4 / 1.5 / 2.1
  - 核心：把图像压缩到 Latent Space，再用 U-Net 去噪。
- 2023：大分辨率与可控生成
  - SDXL、`ControlNet`、GLIGEN、IP-Adapter
  - 核心：提升画质，并且让生成过程能够接受结构、位置和参考图控制。
- 2024：Diffusion Transformer（DiT）兴起
  - SD3、SD3.5、FLUX.1、PixArt、Sana
  - 核心：`Transformer 逐渐替代传统 U-Net`，扩展模型规模和文本理解能力。
- 2025：DiT 与 Flow Matching 成为重要主流路线
  - Qwen-Image、FLUX.2、Z-Image、HunyuanImage、OmniGen2、Janus-Pro
  - 核心：更好的文字渲染、多图参考、图像编辑、统一多模态生成。
- 2026：高效化、统一生成编辑、多模态架构
  - Qwen-Image 新版本、FLUX.2 系列、Z-Image 系列，以及自回归和混合式架构。
  - 核心：生成、编辑、理解、参考图控制逐渐融合。


SD: 一张图片是如何从随机噪声生成的。


Lora,gligen: 如何控制生成内容、风格、位置和结构。


-------------


为什么新模型开始采用 Transformer？

- u-net 本质是一个卷积神经网络CNN, 主要关注局部领域，对多个物体之间复杂关系不友好。
- 扩展到多模态时，如理解空间布局、多种参考图片、视频一致性时，需要 u-net 不断注入模块


