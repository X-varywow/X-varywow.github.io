
```python
params = {'n_vocab': len(train.unique_tokens),
          'embedding_size': 100,
          'num_layers': 2,
          'dropout': 0.5,
          'lr': 1e-3}

model = LSTM1(**params)
```

## 神经网络


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Returns a new tensor with a dimension of size one inserted at the specified position.
# 1 x 100
x = torch.unsqueeze(torch.linspace(-1,1,100), dim = 1)
# print(x)
# add random noise
y = x.pow(3) + 0.1*torch.randn(x.size())


# 将数据转化成Variable的类型用于输入神经网络
x , y =(Variable(x),Variable(y))
plt.scatter(x.data,y.data)
plt.show()

# way 2:
# plt.scatter(x.data.numpy(),y.data.numpy())
```


```python
# 搭建神经网络
class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)

        return out

net = Net(1,20,1)
print(net)
```

```python
# 构建优化目标及损失函数
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
loss_func = torch.nn.MSELoss()

for t in range(1, 5001):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    # loss 反向传播
    loss.backward()
    # 对梯度优化
    optimizer.step()

    if t%1000 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.05)

net.forward(torch.tensor([4], dtype=torch.float))
```


## 查看 pth 文件

```python
d = torch.load("pretrained_ljs.pth")
print(d.keys())
print(list(d['model'].keys())[:3], len(d['model'].keys()))
print(d['model']['enc_p.emb.weight'])
print(type(d['model']['enc_p.emb.weight']), d['model']['enc_p.emb.weight'].dtype)
```

## 划分数据集

https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

```python
dataset = [str(i) for i in RESAMPLED_AUDIO.glob("*.wav")]
n = len(dataset)
num1 = int(n*0.8)
num2 = int(n*0.15)
num3 = n - num1 - num2

train_dataset, test_dataset, val_dataset = random_split(
    dataset = dataset,
    lengths = [num1, num2, num3],
    generator = torch.Generator().manual_seed(42)
)

# *zip([train_dataset, test_dataset, val_dataset], ["train","test","val"])
# SyntaxError: can't use starred expression here ???

for idx in range(3):
    data = [train_dataset, test_dataset, val_dataset][idx]
    desc = ["train","test","val"][idx]
    for file in tqdm(data, desc=desc):
        _, filename = file.split("/")
        filename = filename[:-4]
        
        with open(f"./filelists/s5_{desc}_v1.txt", "a") as f:
            with open(f"./{TXT_PATH}/{filename}.txt", "r") as txt_in:
                f.write("|".join([file, txt_in.read()]))
```

[pytorch 保存模型+加载模型+修改部分层+冻结部分层+删除部分层](https://blog.csdn.net/qq_33328642/article/details/120990405)