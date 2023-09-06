
tqdm -- 进度条显示

（1）基本用法

```python
from tqdm import tqdm
import time
 
for i in tqdm(range(100), desc="progress:"):
    time.sleep(0.1)
    pass
```

（2）利用 pbar 手动更新

```python
from tqdm import tqdm
import time

with tqdm(total=100) as pbar:
    for i in range(100):
        time.sleep(0.05)
        pbar.update(1)
```

(3) 机器学习中使用，[参考](https://github.com/williamyang1991/DualStyleGAN)

```python
from tqdm import tqdm

pbar = tqdm(range(args.iter))
for i in pbar:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_description(
        (
            f"[{ii:03d}/{len(files):03d}]"
            f" Lperc: {Lperc.item():.3f}; Lnoise: {Lnoise.item():.3f};"
            f" LID: {LID.item():.3f}; Lreg: {Lreg.item():.3f}; lr: {lr:.3f}"
        )
    )
```



----------------------
参考资料：
- [详细介绍Python进度条tqdm的使用](https://www.jb51.net/article/166648.htm) 
