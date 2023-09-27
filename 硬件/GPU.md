

GPU包括更多的运算核心，但没有逻辑处理核心，因此特别适合数据并行的计算密集型任务，如大型矩阵运算

深度学习如何选择GPU：
- GPU 架构
- CUDA 核心数量
- 显存带宽（GPU每秒能从GPU显存中读取的数据大小）= 工作频率 x 显存位宽/8bit


[RTX4090 VS Tesla-V100](https://www.topcpu.net/gpu-c/GeForce-RTX-4090-vs-NVIDIA-Tesla-V100-PCIe-32-GB)

4090大致是V100 3倍性能？

CPU、GPU 性能排行榜：https://www.topcpu.net/

|      |       |         |     |
| ---- | ----- | ------- | --- |
| 4090 | 1599$ | 11487   |     |
| 4080 | 1199$ | 8613.50 |     |


CUDA 参考 ML/common/CUDA


## _DLSS_


DLSS3 (40系显卡额外的功能)

DLSS3会分析 GeForce RTX 40 系列 GPU 中全新光流加速器的连续帧和运动数据来生成其它高质量帧，且不影响画质和响应速度。

Deep Learning Super Sampling: AI-Powered Performance Multiplier Boosts Frame Rates By Up To 4X


--------------

支持 DLSS 3.5 和全景光线追踪的 赛博朋克2077：往日之影， `2023.09.27`

很少有新技术首先用在游戏上的，，


光线追踪，采用了递归式光线追踪，指定光线的递归层数，光线在场景中多次碰撞，实时追踪渲染。这比传统的光线投射更准确、拟真。


3.5 除了继续优化 DLSS 深度学习超采样，DLAA 深度学习抗锯齿等技术，又引入了新特性：光线重构，主要解决实时光线追踪中出现的欠采样问题，核心在于对降噪器的替换。


之前的人工降噪器采用 启发式算法，一种基于经验的构造，深度学习降噪器可以识别图像的特征，自动生成光照效果，光流加速器，光流场。




----------

参考资料：
- [GPU为什么这么高效](https://mp.weixin.qq.com/s/jK1sa4zTRWvOjMJzZV4JOA)
- [公版显卡和非公版显卡有啥区别](https://zhuanlan.zhihu.com/p/45816942)
- https://developer.nvidia.cn/zh-cn/rtx/dlss
- [NVIDIA DLSS 3](https://www.nvidia.com/en-us/geforce/news/dlss3-ai-powered-neural-graphics-innovations/)
- [如何评价英伟达推出的 DLSS3.5?](https://www.zhihu.com/question/618638060)
