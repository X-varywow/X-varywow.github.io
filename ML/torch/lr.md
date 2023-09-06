
## _学习策略_

```python
from torch.optim import lr_scheduler
```

- lr_scheduler.StepLR
  -  等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size。
- lr_scheduler.MultiStepLR
- lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=- 1, verbose=False)
  - 指数衰减调整学习率，调整公式: lr = lr * gamma**epoch
- lr_scheduler.CosineAnnealingLR
- lr_scheduler.ReduceLROnPlateau
- lr_scheduler.LambdaLR