
Python 数据处理首页

>可视化参考路线：Python 进行初步的数据处理，并将需要的分析数据处理好，导出到 echarts，用 echarts 作图。

使用 streamlit 制作数据看板类的网页，十分方便。

-------------

使用 ydata profiling 生成数据的详细报告（不适合大量的数据）


```python
# !pip install ydata-profiling

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])

profile = ProfileReport(df, title="Profiling Report")

# 生成 notebook 内分析报告
profile.to_notebook_iframe()
```




-------------

关于 `GPU加速`

使用 [cuDF - GPU DataFrame Library](https://github.com/rapidsai/cudf)

使用 [cuML - GPU Machine Learning Algorithms](https://github.com/rapidsai/cuml)

-------------

关于 `protobuf`

google 开发的一套用于数据存储，网络通信时用于协议编解码的工具库。

是一种二进制的数据格式，相比 XML 和 JSON 具有更高的传输、打包和解包速率

好多次报错要降版本:

```bash
pip uninstall protobuf
pip install protobuf==3.20.3
```


-------------


参考资料：
- [TMDb电影数据分析 & 电影评分预测](https://www.jianshu.com/p/9d7d56dadcc6)
- [IMDB高票房电影数据分析](https://www.jianshu.com/p/a1fee4b3b5b1)
- [(大)数据分析：豆瓣电影分析报告【1】](https://www.jianshu.com/p/9cd6d73a7a62)
