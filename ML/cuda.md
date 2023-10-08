

## _lib_

|                                    |                                    |
| ---------------------------------- | ---------------------------------- |
| libcuda.so                         | The NVIDIA CUDA Driver Library     |
| libcudart.so                       | The NVIDIA CUDA Runtime Library    |
| libcublas.so                       | The NVIDIA cuBLAS Library          |
| libcusparse.so                     | The NVIDIA cuSPARSE Library        |
| libcusolver.so                     | The NVIDIA cuSOLVER Library        |
| libcufft.so, libcufftw.so          | The NVIDIA cuFFT Libraries         |
| libcurand.so                       | The NVIDIA cuRAND Library          |
| libnppc.so, libnppi.so, libnpps.so | The NVIDIA CUDA NPP Libraries      |
| libnvvm.so                         | The NVIDIA NVVM Library            |
| libdevice.so                       | The NVIDIA libdevice Library       |
| libcuinj32.so, libcuinj64.so       | The NVIDIA CUINJ Libraries         |
| libnvToolsExt.so                   | The NVIDIA Tools Extension Library |


</br>

**架构相关：**

`2020.09`：NIVIDIA 发布了基于 AMPERE 架构的显卡3080与3090

- 非安培架构（2080Ti 系列）： CUDA 10.2 性能最优
- 安培架构（3090 系列）： CUDA 11.2 性能更优（ 11.1 和 11.0是过度版本）

安培架构相比图灵架构，引入了第二代RT CORE，第三代Tensor Core


</br>

## _cudnn_

CUDA Deep Neural Network library, 是一个针对深度神经网络的加速库

由 NVIDIA 开发, 通过利用GPU的并行计算能力，提供了一系列高性能的基本操作，如卷积、池化、归一化等，以加速深度神经网络的训练和推断过程。




</br>

## _报错_

（1）

RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


解决方法：export CUDA_LAUNCH_BLOCKING=1 

这样就能跑了，，，




?>CUDA_LAUNCH_BLOCKING是一个环境变量，用于控制CUDA函数在主机线程中执行还是异步执行。</br></br>
当CUDA_LAUNCH_BLOCKING=1时，CUDA函数将在主机线程中进行同步执行。这意味着主机线程将一直等待CUDA函数执行完成，然后再继续执行下一行代码。这种模式对于调试CUDA代码非常有用，因为它可以确保主机线程在CUDA函数执行完成之前不会继续执行其他代码。</br></br>
当CUDA_LAUNCH_BLOCKING=0时，CUDA函数将在主机线程中进行异步执行。这意味着主机线程不会等待CUDA函数执行完成，而是立即继续执行下一行代码。这种模式对于需要在CUDA函数执行期间执行其他任务的情况非常有用。</br></br></br>
默认情况下，CUDA_LAUNCH_BLOCKING的值为0，即异步执行模式。但是在某些情况下，例如在调试代码时，使用同步执行模式可能更方便。可以通过在命令行中设置CUDA_LAUNCH_BLOCKING的值来更改该行为，或者在程序中使用 `cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 1)` 来将其设置为同步执行模式。





-----------

参考资料：
- https://developer.nvidia.com/zh-cn/blog/cuda-intro-cn/
- https://helpmanual.io/man7/libcurand.so/
- [CUDA兼容性问题（显卡驱动、docker内CUDA）](https://zhuanlan.zhihu.com/p/459431437)
- [显卡GPU架构介绍之-----Ampere（安培）](https://www.zhihu.com/tardis/zm/art/395847769)
- [NVIDIA 游戏技术](https://developer.nvidia.cn/zh-cn/industries/game-development)
- chatgpt