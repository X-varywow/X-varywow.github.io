﻿
> 用 ai 来编码，这种逻辑清晰的工作。

### _简单示例_


（1）定义数据

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义 x, y 
x = np.linspace(900, 2100, 1000)
y = 1.0 / (1 + 10 ** ((x - 1500) / 400))

# 定义方式二
x = np.arange(1,11)
y = 2*x+5

```



（2）画图
```python
plt.plot(x, y)
plt.xlabel('combat power')
plt.ylabel('except win rate')
plt.title('VS 1500')

plt.xticks(np.arange(900, 2200, 100))
plt.yticks(np.arange(0, 1.0, 0.1))
plt.show()
```

（3）更多操作

```python
import datetime

# 绘制垂线
plt.axvline(x=datetime.date(2023, 9, 17), color='r', linestyle='--')
# 绘制水平线
plt.axhline(y=0.8, color='r', linestyle='--')

# 添加标签
plt.plot(..., label="name")
plt.legend()

# 旋转 x 坐标
plt.xticks(rotation=-60)

# 绘制网格
plt.grid(True)

# 保存图片，默认 png； bbox_inches 参数去掉白边
plt.savefig('foo', dpi=600, bbox_inches='tight')


# 调高分辨率，将默认的 100dpi 调为 300dpi
# 需放在 plot() 前面
plt.figure(dpi=300)
```

字体

```python
# 字体设置，中文乱码
plt.rcParams['font.family']=['Microsoft YaHei']
## setting global settings
plt.rcParams.update({'font.size': 10,'lines.linewidth': 3})

# 查看字体
from matplotlib import font_manager
sorted([f.name for f in font_manager.fontManager.ttflist])

# f.fname 可以查看路径

plt.rc('font',family='Nimbus Roman') 

# sagemaker 比较好看的字体
plt.rcParams['font.family'] = 'DejaVu Sans'


# win 端比较好看的字体
plt.rcParams['font.family']=['Bookman Old Style']
```

DEMO1: 子图绘制并调整长宽

```python
import numpy as np
import matplotlib.pyplot as plt

# 创建一个包含x值的数组
x = np.linspace(-10, 10, 100)

# 计算sigmoid函数的值
sigmoid = 1 / (1 + np.exp(-x))

# 计算tanh函数的值
tanh = np.tanh(x)

# 创建一个1行2列的subplot，第一个subplot用于绘制sigmoid函数，第二个subplot用于绘制tanh函数
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# 绘制sigmoid函数
axs[0].plot(x, sigmoid)
axs[0].set_title('Sigmoid')

# 绘制tanh函数
axs[1].plot(x, tanh)
axs[1].set_title('Tanh')

# wspace 调整子图之间宽度间距
# hsapce 调整子图之间高度间距
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 显示图形
plt.show()
```



DEMO2: 桌球路线绘图
```python
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

# 为【init, target, hit, final, cushion】 为元素组成的列表
res_x = []
res_y = []

strength = []

for v in d_nxt.values():
    if v['current_router']['strength']>500 and v['after_result']['obs']['summary']['magazine'] <=1:
        init = v['current_router']['white_ball']['ball_coordinate']
        target = v['current_router']['target_ball']['ball_coordinate']
        hit = v['current_router']['hit_ball_point']
        final = v['after_result']['obs']['balls']['0']
        cushion = v['current_router']['fail_pole_spin']['white_cushion_intersection']
        
        tmpx = [init['x'], target['x'], hit['x'], final['x'],cushion['x']]
        tmpy = [init['y'], target['y'], hit['y'], final['y'],cushion['y']]
        
        res_x.append(tmpx)
        res_y.append(tmpy)
        strength.append(v['current_router']['strength'])

def get_data(idx):
    
    x = res_x[idx]
    y = res_y[idx]
    num = strength[idx]
    return x,y,num
# print(x,y,num)


x,y,num = get_data(2)

for i,label in enumerate(['i','t','h','f','c']):
    plt.annotate(label, (x[i], y[i]))

plt.scatter(daizi_x, daizi_y)
plt.scatter(x, y)

plt.plot([x[0], x[2], x[4], x[3]], [y[0],y[2],y[4], y[3]])

plt.title(num)
plt.show()
```


## _subplots_

[官方文档](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)

```python
# using the variable ax for single a Axes
fig, ax = plt.subplots()

# using the variable axs for multiple Axes
fig, axs = plt.subplots(2, 2)

# using tuple unpacking for multiple Axes
fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
```



```python
# 热度图与散点图实例
data = []
f, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 6))

sns.heatmap(data, fmt='d', cmap='YlGnBu',ax=ax1)
for ball in balls:
    if ball["pocket-index"] == -1:
        ax2.scatter(ball["x"], ball["y"])
        ax2.annotate(ball["number"], (ball["x"], ball["y"]))
```

简单的写法：

```python
plt.figure(figsize=(10,5))

plt.subplot(1,2,1) # 1row, 2col, fig1
df[df['LABEL'] == 0]['SCORE'].hist(bins=100)

plt.subplot(1,2,2)
df[df['LABEL'] == 1]['SCORE'].hist(bins=100)
```






</br>

## _更多图形_

`plt.plot(x,y,format_string,**kwargs)`
- `format_string`,控制曲线的格式字符串，可选
- `**kwargs`,第二组或更多`(x,y,format_string)`，可选

**format_string**
1. 颜色字符
2. 风格字符
`'-'`，实线
`'--'`，破折线
`'-.'`，点划线
`':'`，虚线
`'.'`，点标记
`','`，像素标记
`'o'`，实心圆标记
`'v'` `'^'`，三角标记
`'x'`,`'*'`


| 画图方法         | 说明             |
| ---------------- | ---------------- |
| `plt.plot()`     | 绘制坐标图       |
| `plt.boxplot()`  | 绘制箱型图       |
| `plt.scatter()`  | 绘制 **散点图**  |
| `plt.bar()`      | 绘制 **条形图**  |
| `plt.barh()`     | 绘制横向条形图   |
| `plt.hist()`     | 绘制直方图       |
| `plt.polar()`    | 绘制极坐标图     |
| `plt.pie()`      | 绘制饼图         |
| `plt.psd()`      | 绘制功率谱密度图 |
| `plt.specgram()` | 绘制谱图         |
| `plt.cohere()`   | 绘制相关性函数   |
| `plt.step()`     | 绘制步阶图       |


</br>

## _other_



设置坐标轴：
```python
from matplotlib import ticker
fig, ax = plt.subplots()

# 坐标轴数字的间距为 1
locator = ticker.MultipleLocator(1)
ax.xaxis.set_major_locator(locator)

# 坐标轴数字格式为百分式
formatter = ticker.PercentFormatter(xmax=5)
ax.xaxis.set_major_formatter(formatter)
```




```python
#感知机中画图
def show_model(model):
    x_ = np.linspace(4, 7, 100)
    y_ = -(model.w[0] * x_ + model.b)/model.w[1]
    plt.scatter([i[0] for i in X[:50]], [i[1] for i in X[:50]], label='0')  # scatter散点
    plt.scatter([i[0] for i in X[50:]], [i[1] for i in X[50:]], label='1')
    plt.plot(x_, y_)
    plt.title("After")
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()  #可画出存在的图
```


```python
%matplotlib inline

# 使用这个是 Ipython 的内嵌画图，可以省去 plt.show
```


样例：球桌评分热度图 & 球桌球形图
```python
def vis(self):
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = []

    for j in range(55, -56, -STEP):
        tmp = []
        for i in range(-113, 114, STEP):
            x = COL_IDX[i]
            y = ROW_IDX[j]
            tmp.append(self.mat[y][x])
        data.append(tmp)

    # Draw a heatmap with the numeric values in each cell
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 6))
    ax1.set_xlim(-113,113)
    ax1.set_ylim(-55,55)
    ax2.set_xlim(-113,113)
    ax2.set_ylim(-55,55)
    sns.heatmap(data, fmt='d', cmap='YlGnBu',ax=ax1)
    
    
    x1 = []
    x2 = []
    x3 = []
    y1 = []
    y2 = []
    y3 = []
    for ball in self.status['balls']:
        if ball["pocket-index"] == -1:
            if ball["number"] in range(1,8):
                x1.append(ball["x"])
                y1.append(ball["y"])
            elif ball["number"] in range(9,16):
                x2.append(ball["x"])
                y2.append(ball["y"])
            else:
                x3.append(ball["x"])
                y3.append(ball["y"])
    
    ax2.scatter(x1, y1)
    ax2.scatter(x2, y2)
    ax2.scatter(x3, y3)
    
    for ball in self.status['balls']:
        if ball["pocket-index"] == -1:
            ax2.annotate(ball["number"], (ball["x"], ball["y"]))
            
    plt.show()
```


--------------

设置后端

Agg后端（非交互式，只生成静态图像，完全线程安全，可在任何线程使用）；使用场景：服务器端、后台任务、无GUI环境

```python
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前调用
import matplotlib.pyplot as plt
```






------------

参考资料：
- [官方文档](https://matplotlib.org/stable/contents.html)
- [官方示例](https://matplotlib.org/stable/gallery/index.html)
- [官方 cheatsheet](https://matplotlib.org/cheatsheets/)
- [pyplot turtorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)