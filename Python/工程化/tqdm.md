
## _tqdm_

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

参考：[详细介绍Python进度条tqdm的使用](https://www.jb51.net/article/166648.htm) 

</br>

## _rich_

参考：https://rich.readthedocs.io/en/stable/progress.html


```bash
pip install rich
```

Basic usage:

```python
import time
from rich.progress import track

for i in track(range(20), description="Processing..."):
    time.sleep(1)  # Simulate work being done
```

Advanced usage:

```python
import time

from rich.progress import Progress

with Progress() as progress:

    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)

    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        time.sleep(0.02)
```

显示效果：

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231102174750.png">

</br>

demo1: so-vits-svc 中使用

```python
def process_all_speakers():
    process_count = 30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                print(spk_dir)
                futures = [executor.submit(process, (spk_dir, i, args)) for i in os.listdir(spk_dir) if i.endswith("wav")]
                for _ in track(concurrent.futures.as_completed(futures), total=len(futures), description="resampling:"):
                    pass
```

</br>

## _rich 其它用法_

相当于给 普通IO 套了一层 UI

参考：https://rich.readthedocs.io/en/stable/introduction.html

```python
from rich import print
from rich.panel import Panel
print(Panel.fit("Hello, [red]World!"))
```

```python
import time

from rich.live import Live
from rich.table import Table

table = Table()
table.add_column("Row ID")
table.add_column("Description")
table.add_column("Level")

with Live(table, refresh_per_second=4):  # update 4 times a second to feel fluid
    for row in range(12):
        time.sleep(0.4)  # arbitrary delay
        # update the renderable internally
        table.add_row(f"{row}", f"description {row}", "[red]ERROR")
```