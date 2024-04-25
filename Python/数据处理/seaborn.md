## _直线图_

```python
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()

sns.lineplot(data=data, palette="tab10", linewidth=2.5)
```


</br>

## _直方图_


```python
import seaborn as sns
sns.histplot(df[df['NEXT_ENTRY_KIND'] == 2]['NEXT_SCORE'], kde=True)

# 直方图+核密度 (deprecated)
sns.distplot() 

# kernel density estimate，核密度估计
sns.kdeplot(data=data['column_name'])
# 双变量 KDE
sns.kdeplot(data=data, x='feature1', y='feature2', shade=True)
```

</br>

## _气泡图_


```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 假设df是您的DataFrame

# 修改数据，增加一列作为气泡的大小
df['size'] = np.log(df['col2'])

# 使用seaborn的scatterplot函数绘制气泡图，并设置透明度
# sizes 表示：线性映射的最小值、最大值
sns.scatterplot(data=df, x='col1', y='col1', size='size', legend=False, sizes=(20, 2000), alpha=0.5)

# 通过matplotlib加以定制，比如添加标题等
plt.title('Bubble Plot with Transparency')
plt.show()
```

</br>

## _热度图_

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

data = [[1,2],[2,3]]

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(data, fmt='d', linewidths=.5, cmap='YlGnBu')
```








---------

参考资料：
- [seaborn_gallery](https://seaborn.pydata.org/examples/index.html)
- chatgpt