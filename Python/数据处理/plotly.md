
https://github.com/plotly/plotly.py


3D 还挺好的：

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

还是 streamlit 吧：

https://github.com/streamlit/streamlit