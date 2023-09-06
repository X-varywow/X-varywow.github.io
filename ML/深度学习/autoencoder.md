

## _1.preface_


AE 自编码器，无监督式学习模型

[【深度学习】 自编码器（AutoEncoder）](https://zhuanlan.zhihu.com/p/133207206)


MAE 介绍：[如何看待何恺明最新一作论文Masked Autoencoders？](https://www.zhihu.com/question/498364155/answer/2240224120)



</br>

## _2.代码实例_

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义自动编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 加载数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in trainloader:
        images, _ = data
        images = images.view(images.size(0), -1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(trainloader)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, avg_loss))

# 使用自动编码器进行图像重建
import matplotlib.pyplot as plt

# 随机选择一批图像
dataiter = iter(trainloader)
images, _ = dataiter.next()

# 将图像输入自动编码器进行重建
images = images.view(images.size(0), -1)
reconstructed = model(images)

# 显示原始图像和重建图像
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
for images, row in zip([images, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.view(28, 28).numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
```

在这个示例中，我们定义了一个简单的自动编码器模型，用于对MNIST数据集中的手写数字图像进行编码和解码。模型包含一个编码器和一个解码器，它们都是由全连接层组成的简单神经网络。

我们使用MNIST数据集进行训练，然后使用自动编码器对一批图像进行重建。最后，我们使用matplotlib库将原始图像和重建图像进行可视化。


----------

参考资料：
- chatgpt