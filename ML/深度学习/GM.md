

> 主流的的生成模型：VAE, GAN, flow
- GAN
    - a generative model `G` that captures the data distribution
    - a discriminative model `D` that estimates the probability that a sample came from the training data rather than G
- VAE
- Flow


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230822004226.png" style="zoom:60%">

</br>

## _GAN_

Generative Adversarial Nets，生成对抗网络

原文地址：https://arxiv.org/abs/1406.2661

原理：生成器、判别器两者不断影响，目标是利用生成的数据替代真实的数据。最终使用生成器做生成就行了。

-----------

GAN 有着非常多的变种：DCGAN、WGAN 等


（DCGAN, pytorch 版）代码样例：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

torch.manual_seed(1234)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入为 nz 维的随机向量
            nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # (ngf*4)x4x4
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2)x8x8
            nn.Convtranspose2d(ngf*2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngfx16x16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
            # nc x 32 x 32
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input: nc x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input).view(-1, 1)    
```

```python
# 超参数设置
batch_size = 128
lr = 0.0002
beta1 = 0.5  # Adam优化器的beta1参数

nz = 100  # 随机噪声维度
ngf = 64  # 生成器的feature map深度
ndf = 64  # 判别器的feature map深度
nc = 1  # 图片通道数

#数据载入
transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,)),
           ])
dataset = MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建生成器和判别器实例
G = Generator(nz, ngf, nc)
D = Discriminator(nc, ndf)

# 判别器和生成器优化器
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# 损失函数
criterion = nn.BCELoss()

# 训练
num_epochs = 5 
real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ###############
        # 更新判别器D
        ###############
        # 使用全部真实图片
        D.zero_grad()
        real_data = data[0].to(torch.float)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), real_label).to(torch.float)

        output = D(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 使用全部假图片
        noise = torch.randn(batch_size, nz, 1, 1)
        fake_data = G(noise)
        label.fill_(fake_label)
        output = D(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # 更新判别器D的参数
        errD = errD_real + errD_fake
        optimizerD.step()

        ###############
        # 更新生成器G
        ###############
        G.zero_grad()
        label.fill_(real_label) 
        output = D(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        # 更新生成器G的参数
        optimizerG.step()

        # 打印训练状态
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# 最后一步记得保存模型
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')

```







</br>

## _VAE_

变分自编码机(Variational AutoEncoder, VAE)，VAE作为可以和GAN比肩的生成模型，融合了贝叶斯方法和深度学习的优势，拥有优雅的数学基础和简单易懂的架构以及令人满意的性能，其能提取disentangled latent variable的特性也使得它比一般的生成模型具有更广泛的意义。

请参考：[ML/高级/vae](/ML/高级/vae)


</br>

## _Diffusion_

属于无监督生成模型

如 Stable Diffusion (2022 年的深度学习文本到图像生成模型), [wiki](https://zh.wikipedia.org/zh-cn/Stable_Diffusion)。

[大一统视角理解扩散模型](https://zhuanlan.zhihu.com/p/558937247)

[Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)

[Lil'Log - What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---------
参考资料：
- [万字详解什么是生成对抗网络GAN](https://bbs.huaweicloud.com/blogs/314916) ⭐️
- [GAN论文逐段精读【论文精读】](https://www.youtube.com/watch?v=g_0HtlrLiDo)
- [GAN-生成对抗网络原理及代码解析](https://www.bilibili.com/video/BV1ht411c79k/)
- chatgpt