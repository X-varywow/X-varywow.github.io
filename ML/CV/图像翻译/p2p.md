

## preface

图像翻译的经典模型：pix2pix, pix2pixHD, vid2vid


生成器优化目标：
- 对抗损失，建模高频信息
- 重建损失，L1距离，建模低频信息




PPGAN 容易发生特征纠缠，无法精确控制图片生成过程中每一级的特征


卡通图像往往有清晰的边缘，平滑的色块和经过简化的纹理，与其他艺术风格有很大区别。使用传统图像处理技术生成的卡通图无法自适应地处理复杂的光照和纹理，效果较差；基于风格迁移的方法无法对细节进行准确地勾勒。



vid2vid face-edge-face 挺有用的


新的模型：[instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix)


## _1. 基础组件_

### 1.1 U-Net

全卷积结构（只使用卷积层和池化层，没有全连接层）（使输出结果仍是一个特征图或特征向量，而不是分类标签或回归值）

，用于处理任意尺寸的输入数据，并保留输入数据的空间结构信息。

加入skip-connection，对应的feature maps和decode之后的同样大小的feature maps按通道拼(concatenate)一起，用来保留不同分辨率下像素级的细节信息。



### 1.2 PatchGAN



### 1.3 cGAN


## _2. 一些理解_

pix2pix是基于cGAN实现图像翻译，通过添加条件信息来指导图像生成，可以将输入图像作为条件，学习从输入图像到输出图像之间的映射，从而得到指定的输出图像。

而其他基于GAN来做图像翻译的，因为GAN算法的生成器是基于一个随机噪声生成图像，难以控制输出

训练时需要成对的图像


## _3. 代码实现_



----------

参考资料：
- [pix2pix算法笔记](https://mapengsen.blog.csdn.net/article/details/115425377)
- [图像翻译三部曲：pix2pix, pix2pixHD, vid2vid](https://zhuanlan.zhihu.com/p/56808180)
- https://blog.csdn.net/qq_56591814/article/details/125419686
- [github vid2vid](https://github.com/NVIDIA/vid2vid)
- [github photo2cartoon](https://github.com/minivision-ai/photo2cartoon)