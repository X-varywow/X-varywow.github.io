
https://pytorch.org/docs/stable/data.html


A custom collate_fn can be used to customize collation, e.g., padding sequential data to max length of a batch. 


DATASET & DATALOADERS

洗牌，发牌

```python
import os
import pandas as pd
from torchvision.io import read_image

# must implement: init, len, getitem
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# 创建 dataset 实例
dataset = CustomImageDataset(data)
# 创建 dataloader 实例
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    pass

```

collate_fn 负责对一个 batch 做一个后处理；如 padding 操作
collate: collect and combine 

dataset types:
- map-style 通用
- iterable-style , 一般用于流式

eval() set the module evaluation mode. 影响 dropout, batchnorm 层

load_state_dict() 保存模型参数


state_dict() 返回模型字典

```python
import torch

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear1 = torch.nn.Linear(2,3)
        self.linear2 = torch.nn.Linear(3,4)
        self.bn = torch.nn.BatchNorm2d(4)
        
test = Test()

test._modules
# OrderedDict([('linear1', Linear(in_features=2, out_features=3, bias=True)),
#              ('linear2', Linear(in_features=3, out_features=4, bias=True)),
#              ('bn',
#               BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))])

test._modules['linear1'].weight
test._modules['linear1'].weight.dtype

test.state_dict()
# OrderedDict([('linear1.weight',
#               tensor([[-0.1885, -0.5291],
#                       [-0.7071, -0.2659],
#                       [-0.5842, -0.3362]])),
#              ('linear1.bias', tensor([0.3272, 0.5955, 0.6850])),
#              ('linear2.weight',
#               tensor([[-0.4601, -0.4413,  0.5047],
#                       [-0.1733, -0.4730, -0.3636],
#                       [-0.4972, -0.1445, -0.2071],
#                       [ 0.3664, -0.4267,  0.1859]])),
#              ('linear2.bias', tensor([-0.5159, -0.0586, -0.4024, -0.5557])),
#              ('bn.weight', tensor([1., 1., 1., 1.])),
#              ('bn.bias', tensor([0., 0., 0., 0.])),
#              ('bn.running_mean', tensor([0., 0., 0., 0.])),
#              ('bn.running_var', tensor([1., 1., 1., 1.])),
#              ('bn.num_batches_tracked', tensor(0))])
```