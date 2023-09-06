

仓库地址：https://github.com/iperov/DeepFaceLab

（2021）论文地址：https://arxiv.org/abs/2005.05535


基于 autoencoder 的 SOTA

3000 ~ 8000 张，尽量覆盖不同的人脸角度、光照条件、面部表情。

[DFL 介绍](https://zhuanlan.zhihu.com/p/140444440) 评论区，感觉好难。


但是整个脸的形变还是不行，，，大体框架几乎定下来了，顶多进行二次转换


## 问题

需要训练较长时间

贴合可能存在明显的边幅问题，比如肤色不同时，就很明显



## 说明

工作流：
- 提取
  - extract frame
  - face detection
  - face alignment
  - face segmentation
- 训练
- 转换


- DF 结构
  - 训练共享权重的 Encoder 和 Inter，不同的 decoder
- LIAE 结构
  - encoder decoder 共享权重， interAB interB 独立


[deepfacelab 视频教程](https://www.youtube.com/watch?v=kOIMXt8KK8M)

生成 xseg mask

视频中，两个 30s 的视频训练 12h

如何提高收敛速度？

这个融合怎么对齐的，参考一下


https://github.com/chervonij/DFL-Colab/blob/master/DFL_Colab.ipynb


------------------

参考资料：
- https://www.youtube.com/watch?v=pOv6SsnUFbU
- https://www.youtube.com/watch?v=kOIMXt8KK8M
- DFL 社区：https://dfldata.cc/