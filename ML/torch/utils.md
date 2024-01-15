

1. `torch.utils.data`
   - `DataLoader`：用于封装可迭代的数据加载器，自动化数据的批处理、打乱等。
   - `Dataset`：用于定义新的数据集，需要自定义`__getitem__`和`__len__`方法。
   - `TensorDataset`：包装张量数据的数据集，实现索引和长度方法。
   - `Subset`：用于选择原有数据集的一个子集。
   - `random_split`：用于随机划分数据集为非重叠的新数据集群组。

2. `torch.utils.model_zoo`
   - 此模块在新版本的PyTorch中被弃用，模型载入和存储的功能被整合到`torchvision`的`models`模块之中。

3. `torch.utils.tensorboard`
   - PyTorch的TensorBoard支持模块，可以使用`SummaryWriter`类记录和可视化模型训练过程。

4. `torch.utils.checkpoint`
   - `checkpoint`和`checkpoint_sequential`：提供了梯度checkpoint的功能，用于训练时内存的优化。
   
5. `torch.utils.data.sampler`
   - 包含用于数据加载的各种采样器，如`WeightedRandomSampler`、`RandomSampler`、`SequentialSampler`等，它们规定数据的读取方式。
  
6. `torch.utils.data.dataloader`
   - 已构建于`torch.utils.data`模块中，提供`DataLoader`类的实现。
  
7. `torch.utils.benchmark`
   - 提供基准测试工具，可用来衡量和比较代码性能。