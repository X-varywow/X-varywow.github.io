

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


--------------

相比同构图，**异构图** 里可以有不同类型的节点和边。比如在IMDB中，可以有三类node分别是Movie，Director和Actor

**二分图**，又叫二部图。每条边两个顶点属于唯二的顶点集。

GNN 相对于传统的神经网络（如经典的 Embedding+MLP 架构，RNN 等），都是通过 NN 拟合输入输出之间的关系，GNN **能适应更加复杂的结构性先验**，描述复杂的非线性结构。


?> 图神经网络和之前的 DeepWalk，Node2vec 等 Graph Embedding 方法有什么不同？</br></br>
<u>基于 Random Walk 的 Graph embedding 方法本质上没有直接处理图结构</u>, 而是通过将节点和邻域序列化, 转化为类似于文本的线性结构。</br></br>
图神经网络的优势在于能够直接处理图结构, 同时进行节点和邻域之间的信息传播以及参数更新。<u>更加端到端，更加通用</u>。


GNN 适用性更强，在大部分的应用场景下，效果不会太差，而且更加鲁棒。

如果数据中序列性非常强，或者要研究的问题跟时间强相关，我个人觉得直接采用序列模型建模更加合适；

<u>如果数据比较稀疏，需要邻域节点做信息协同建模，那基于空间的图神经网络就很适合</u>。



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

参考资料：
- [图神经网络（Graph Neural Networks，GNN）综述](https://zhuanlan.zhihu.com/p/75307407)
- [「AI大咖谈」DLP-KDD最佳论文作者谈「图神经网络」的特点、发展与应用](https://zhuanlan.zhihu.com/p/259494288)
- chatgpt