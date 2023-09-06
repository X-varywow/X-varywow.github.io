
> 自然语言处理作业五：语言模型

## 模型实现

参考资料：
- https://github.com/liux2/RNN-on-wikitext2
- https://paperswithcode.com/sota/language-modelling-on-wikitext-2
- https://github.com/RMichaelSwan/MogrifierLSTM



困惑度：
```python
from torch import Tensor
import numpy as np
import torch.nn.functional as F


def perplexity(outputs: Tensor, targets: Tensor, config=None):
    """
    计算语言模型困惑度
    :param outputs: [batch_size,seq_len,vocab_size]
    :param targets: [batch_size,seq_len]
    :param config:  配置文件 default:None
    :return: 困惑度数值
    """
    ce = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1),
                         ignore_index=config.data.pad_id if config is not None else None)

    return torch.exp(ce)

```

## 常见语言模型

词向量模型 word2vec、glove

离散型 one-hot

基于神经网络的语言模型 

预训练的语言模型：BERT、GPT

参考资料：
- https://zhuanlan.zhihu.com/p/58931044