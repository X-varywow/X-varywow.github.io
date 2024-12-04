
https://github.com/plotly/plotly.py


优点：自动美化坐标轴，支持交互。


## 折线图

可交互查看具体的节点，对应的数据；并且横轴自动美化了；

```python
import plotly.graph_objects as go

x = list(df.loc[:, 't'])
y = list(df.loc[:, 'cnt'])

# 创建折线图
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers', name='Line'))

# 添加标题
fig.update_layout(title='Line Plot with Plotly')

# 显示图形
fig.show()
```


## 3D 图

挺好看的

```python
import plotly.graph_objects as go
import numpy as np

# 生成样本数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# 创建一个 3D 表面图
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='3D Surface Plot', scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Z-axis'))
fig.show()
```