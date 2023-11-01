

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