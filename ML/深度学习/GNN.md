

图神经网络 GNN

主要应用：社交网络分析、推荐系统、生物信息学等


使用工具：
- https://www.dgl.ai/
- https://github.com/pyg-team/pytorch_geometric
- deepwalk


</br>

## _理论基础_

GCN（Graph Convolutional Network）专门用于处理非结构化的图数据，

GCNConv 是模型中的一个核心组件，是种图卷积层；

GCNConv 输入是一个节点的特征矩阵和相邻节点的特征矩阵，输出是该节点的新特征表示

GCNConv 的计算过程可以分为以下几个步骤：

1. 初始化权重矩阵：根据输入特征的维度和输出特征的维度，初始化一个权重矩阵。
2. 归一化邻接矩阵：将图的邻接矩阵进行归一化处理，以便在卷积过程中更好地传播信息。
3. 计算卷积结果：将节点的特征矩阵与归一化后的邻接矩阵相乘，得到节点的新特征表示。
4. 应用非线性激活函数：对卷积结果进行非线性变换，以增加模型的表达能力。







</br>

## _代码_

```bash
pip install torch_geometric
```



```python
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F

dataset = Planetoid(root='.', name='Cora')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义两层 GCN conv 网络的  GCN 网络
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         # x: Node feature matrix of shape [num_nodes, in_channels]
#         # edge_index: Graph connectivity matrix of shape [2, num_edges]
        
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x

    # 加入 dropout 和 log_soft_max
    # 0.0008118064142763615
    # 0.009149790741503239
    # 相比上一个网络，最终误差变大，且中途误差波动变大
    
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GCN(dataset.num_features, 16, dataset.num_classes)


data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
arr = []


for epoch in range(200):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
    arr.append(loss)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```



```python
import matplotlib.pyplot as plt

plt.plot([i for i in range(1,201)], [float(i) for i in arr])

plt.show()

float(arr[-1])
```











------------

参考资料：[图神经网络（Graph Neural Networks，GNN）综述](https://zhuanlan.zhihu.com/p/75307407)