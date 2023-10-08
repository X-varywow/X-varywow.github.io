
## _简单介绍_

VGG， 一种经典的深度神经网络，2014 年提出

在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，




<p class="warn">
深层网络优点：</br>
- 叠加小型滤波器来加深网络可以减少参数的数量，层次越深越明显。</br>
&nbsp; - 如一次 5x5 卷积运算可由 两次 3x3 卷积运算抵充，参数数量 25 减少为 18。</br>
- 扩大感受野（receptive field, 给神经元施加变化的某个局部空间区域）</br>
- 另外，通过叠加层，将激活函数放在卷积层中间，增加非线性特质，提高网络的表现力
</p>


缺点：
- 3层全连接层导致参数过多


</br>

## _代码实现_

手动实现 VGG 网络，达到 1000 分类：

```python
import torch
import torch.nn as nn

# VGG16
class VGG(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = VGG()
print(model)
# 太简洁优雅了，
```

虽然很简洁，但是 VGG 这个自动学习机，一大堆参数训下来，就变成一个十足的黑盒了，




鱼书入门后，觉得VGG网络的积木结构好简洁，就跟车厢连接起来一样。








----------

参考资料：
- [一文读懂VGG网络](https://zhuanlan.zhihu.com/p/41423739)
- [使用Pytorch搭建VGG网络——以VGG11为例](https://www.cnblogs.com/xmd-home/p/14793221.html)
- chatgpt