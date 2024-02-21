

> 主流的的生成模型：VAE, GAN, flow

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230822004226.png" style="zoom:60%">

## _1. 基础组件_

### 1.1 KL散度

Kullback-Leibler divergence, KLD

一种衡量两个概率分布之间差异的度量，用于衡量模型预测分布与真实分布之间的差异。

在信息论中，又称为相对熵：

$$D_{KL}(P||Q) = \sum_{x\in X} P(x)ln(\frac{P(x)}{Q(x)})$$

其中，P 和 Q 为离散概率空间 X 上的两个概率分布。


对于连续型随机变量，则有：$$D_{KL}(P||Q) = \int_{-\infty}^{+\infty} p(x)ln(\frac{p(x)}{q(x)})dx$$

当 P = Q 时，$D_{KL} = 0$。 此度量 **没有对称性**


### 1.2 变分

变分，是研究泛函的极值问题的一个重要数学工具，用于 **研究函数变化与某个参量变化之间关系**

$$I = \int_a^b f(y, y^{\prime})dx$$

---------

希望找到 $y(x)$ ，使得 $I$ 有极值。$I$ 称为 $y(x)$ 的 `泛函`。

如果 $y(x)$ 有微小变化 $\delta y(x)$，$\delta y(x)$ 称为 $y(x)$ 的 `变分`。


------------

$f(y, y^{\prime})$ 的变化为：

$$\delta f = \frac{\partial f}{\partial y} \delta y + \frac{\partial f}{\partial y^{\prime}} \delta y^{\prime}$$


$I$ 的变化为：

$$\delta I = \int_a^b[\frac{\partial f}{\partial y} \delta y + \frac{\partial f}{\partial y^{\prime}} \delta y^{\prime}] dx$$

参考：[浅谈变分原理](https://zhuanlan.zhihu.com/p/139018146)⭐


</br>

## _2. 一些理解_

>简单理解：VAE 从隐藏变量 Z 生成目标数据 x, 将 z 的概况分布尽可能拟合成 x 的模样。

VAE (Variational Autoencoder) 是一种生成模型，它结合了自动编码器（Autoencoder）和变分推断（Variational Inference）的思想。

它通过学习数据的潜在表示，可以用来生成新的样本。

VAE 的基本思想是将输入数据映射到一个潜在空间中，并通过一个解码器将潜在向量重新映射回原始数据空间。

-------------

分为两个阶段：
- 编码阶段
  - 将输入数据通过一个编码器网络映射为均值和方差两个参数，这两个参数用来定义一个潜在空间中的高斯分布。编码器网络的输出可以看作是潜在空间的均值和方差的参数化表示。
- 解码阶段
  - 从潜在空间中采样一个向量，并通过一个解码器网络将其映射回原始数据空间。解码器网络的目标是最大程度地重构原始数据。

为了能够 **更好地还原真实分布**（重构后的分布 $Z$ 尽可能去 拟合真实的先验分布 $X$），VAE 引入了一个重构损失，用来衡量重构数据和原始数据之间的差异。

这个正则化项，即 KL 散度（KL Divergence）。通过最小化 KL 散度，VAE 可以学习到一个在潜在空间中均匀分布的表示。

> 总的来说，VAE 是一种生成模型，通过学习数据的潜在表示，可以生成新的样本。它通过编码器将输入数据映射为潜在空间的分布参数，并通过解码器将潜在向量映射回原始数据空间。通过最小化重构损失和 KL 散度，VAE 可以学习到一个连续且平滑的潜在空间表示。


</br>

## _3. 代码实现_


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encode
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # decode
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    # 编码器，将图像映射为 均值、方差 两个隐变量
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    # 重参数化采样，得到潜在变量 z
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

```python
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# 加载MNIST数据集, 9.5 mb
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

model = VAE()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

epochs = 10
log_interval = 10
for epoch in range(1, epochs + 1):
    train(epoch)
```







-------------

参考资料：

chatgpt

[变分自编码器 = 最小化先验分布 + 最大化互信息](https://spaces.ac.cn/archives/6088)

[变分自编码器（一）：原来是这么一回事](https://spaces.ac.cn/archives/5253)

[变分自编码器（二）：从贝叶斯观点出发](https://spaces.ac.cn/archives/5343)

[变分自编码器（三）：这样做为什么能成？](https://spaces.ac.cn/archives/5383)

[变分自编码器（四）：一步到位的聚类方案](https://spaces.ac.cn/archives/5887)

[变分自编码器（五）：VAE + BN = 更好的VAE](https://spaces.ac.cn/archives/7381)

[变分自编码器（六）：从几何视角来理解VAE的尝试](https://spaces.ac.cn/archives/7725)

[变分自编码器（七）：球面上的VAE（vMF-VAE）](https://spaces.ac.cn/archives/8404)

[变分自编码器（八）：估计样本概率密度](https://spaces.ac.cn/archives/8791)

[细水长flow之f-VAEs：Glow与VAEs的联姻](https://spaces.ac.cn/archives/5977)

[细水长flow之RealNVP与Glow：流模型的传承与升华](https://spaces.ac.cn/archives/5807)

[基于CNN和VAE的作诗机器人：随机成诗](https://spaces.ac.cn/archives/5332)

[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)

[Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

[漂亮国的核潜艇与深度学习的内卷](https://mp.weixin.qq.com/s/NvwaR_dzQZnE85W1YtVWiA) 综述及探讨常用生成模型和分布误差度量方式

对神经网络黑盒的理解：如何对这些信息进行了有效的表征，并且可以泛化；其中数学原理？信息传递方式？压缩即智能？

相变、涌现、很神奇 [学习语言需要相变](https://mp.weixin.qq.com/s/x2mkY3qCJssZ2WJmD0MB_A)

[White-Box Transformers via Sparse Rate Reduction: Compression Is All There Is?](https://arxiv.org/abs/2311.13110) 数学可解释模型 [解读1](https://zhuanlan.zhihu.com/p/661471603)
