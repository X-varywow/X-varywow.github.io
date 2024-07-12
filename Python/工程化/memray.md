

使用 memray 分析实时内存占用（将主体程序抽出来，分析 streamlit 程序有点不好）

github stars: [memray](https://github.com/bloomberg/memray)(11.8K), [py-spy](https://github.com/benfred/py-spy)(11.4k), [scalene](https://github.com/plasma-umass/scalene)(10.6k) memory_profiler(4.1k)

## memray

```bash
pip install memray
memray run tests/test_memory.py

# 实时命令行界面
memray run --live my_script.py

# 在另一个端口打开，本窗口用于观测程序日志
memray run --live-remote ./tests/test_copy.py
memray3.9 live 52614

# 查看分析日志
memray tree
memray flamegraph tests/memray-test_memory.py.12585.bin
```

但是 live 模式只能看内存个大概:
```python
import time
a = []
while 1:
    a += ['abcdefghijklmnopqrstuvwxyz'*1000]*100000
    time.sleep(1)
```
如监听上述文件，只能看到 \<module\> 的占用而看不到详细变量的占用;

代码内部比较细的内存占用，还得看火焰图


## memory_profiler

```python
# pip install memory_profiler

import numpy as np
from memory_profiler import profile
import time

@profile
def demo():
    for i in range(5):
        a = np.random.rand(1000000)
        b = np.random.rand(1000000)
        
        a_ = a[a < b]
        b_ = b[a < b]

        time.sleep(2000)
    
    del a, b

    return a_, b_


if __name__ == '__main__':
    demo()
```


之后测试代码：

```bash
python tests/test_memory.py

# 使用 mprof 查看内存随时间的变化图
mprof run tests/test_memory.py

mprof plot
```

参考资料：https://zhuanlan.zhihu.com/p/121003986


## 内存泄露常见场景


```bash
pip install objgraph
```


