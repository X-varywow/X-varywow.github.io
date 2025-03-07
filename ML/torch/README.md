

学习完 《鱼书》（对神经网络有了解，知道 torch 应该做些什么），再来看 torch 语法


https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html


---------


检查环境：

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
```

---------

安装torch:

(pip 没找到对应的版本， nvidia-smi 可查看本机 cuda 版本)

```bash
conda create -n cu118 -y
conda activate cu118

pip install ipykernel
python -m ipykernel install --user --name=cu118_kernel


conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```






----------

参考资料:
- [视频教程1](https://www.bilibili.com/video/BV1ov411M7xL/)⭐️
- [官网：LEARN THE BASICS](https://pytorch.org/tutorials/beginner/basics/intro.html)⭐️
- [torch 基础语法](https://mp.weixin.qq.com/s/hJBapk-CL3x_c2pXKcJLWw)
- [Pytorch搭建简单神经网络（一）——回归](https://zhuanlan.zhihu.com/p/114980874)
- https://ifwind.github.io/2022/03/20/Pytorch%E6%89%8B%E5%86%8C%E6%B1%87%E6%80%BB/