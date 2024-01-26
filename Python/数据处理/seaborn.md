


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







## _绘制热度图_

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
- 官网