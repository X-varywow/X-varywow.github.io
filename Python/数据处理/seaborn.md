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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建一个示例 DataFrame 数据
data = {
    '年龄': np.random.randint(18, 65, 100),
    '收入': np.random.randint(5000, 20000, 100),
    '消费': np.random.randint(1000, 10000, 100)
}
df = pd.DataFrame(data)

# 计算相关系数矩阵
corr = df.corr()

# 设置 matplotlib 的 figure 大小
plt.figure(figsize=(10, 8))

# 绘制热力图
ax = sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, cmap='YlGnBu')

# 将 x 轴的标签移到上方
ax.xaxis.tick_top()  # x轴刻度线和标签移到上方
ax.xaxis.set_label_position('top')  # x轴标签移到上方

# 可选：旋转标签以改善显示效果
# plt.xticks(rotation=45)  # 旋转 x 轴标签
plt.yticks(rotation=0)  # 旋转 y 轴标签（如果需要）

# 显示图形
plt.show()
```



</br>

## _子图_

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

sns.histplot(df[df['LABEL'] == 0]['SCORE'], kde=True, ax = axs[0])
axs[0].set_title("label = 0")

sns.histplot(df[df['LABEL'] == 1]['SCORE'], kde=True, ax = axs[1])
axs[1].set_title("label = 1")

# plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
# plt.show()
```







---------

参考资料：
- [seaborn_gallery](https://seaborn.pydata.org/examples/index.html)
- chatgpt