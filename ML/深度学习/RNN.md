
<p class="pgreen">RNN, 即带有循环层的网络，能够利用已有的序列信息。以下几种网络都属 RNN 大类</p>


## _RNN_

>循环神经网络, recurrent neural network

RNN 是一种能有效地 **处理序列数据或时序数据** 的神经网络。比如:⽂章内容、语⾳音频、股票价格⾛势。

**主要特点**：每次都会将前一次的输出结果，带到下一次的隐藏层中，一起训练

缺点：无法处理很长的输入序列（长期时间关联），随着递归会带来指数爆炸和梯度消失问题


</br>

## _BRNN_

双向循环神经网络


</br>

## _LSTM_


> 长短期记忆（Long Short-Term Memory，LSTM）是对 RNN 的改进。

1997年提出，2010前后达到大量的 SOTA


由于独特的设计结构，LSTM适合于处理和预测时间序列中间隔和延迟非常长的重要事件。


实现方式1：

```python
from typing import Optional, Tuple
import torch
from torch import nn
from labml_helpers.module import Module


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        super().__init__()
        
        self.hidden_lin = nn.Linear(hidden_size, 4*hidden_size)
        self.input_lin = nn.Linear(input_size, 4*hidden_size, bias = False)

        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        ifgo = ifgo.chunk(4, dim = -1)
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]
        i, f, g, o = ifgo
        
        c_next = torch.sigmoid(f)*c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.cells = nn.MouduleList([LSTMCell(input_size, hidden_size)] + [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])
    
    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        n_steps, batch_size = x.shape[:2]

        if not state:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            h, c = list(torch.unbind(h)), list(torch.unbind(c))

        out = []
        for t in range(n_steps):
            inp = x[t]
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)
```


实现方式2：

```python
from torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```
```python
import torch.optim as optim

# 定义超参数
input_size = 10
hidden_size = 64
num_layers = 1
output_size = 1
learning_rate = 0.001
epochs = 100

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
```



</br>

## _GRU_

GRU 主要是在 LSTM 的模型上做了一些简化和调整，在训练数据集比较大的情况下可以节省很多时间。

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230824181535.png" style="zoom:80%">



---------------


参考资料：
- 粗略地介绍rnn： https://easyai.tech/ai-definition/rnn/
- [RNN（循环神经网络）基础](https://zhuanlan.zhihu.com/p/30844905)
- [WIKI - RNN](https://zh.wikipedia.org/zh-hans/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
- 代码参考：https://nn.labml.ai/lstm/index.html