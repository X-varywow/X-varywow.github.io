
介绍：平面几何分析与计算, [docs](https://shapely.readthedocs.io/en/stable/)

底层基于 GEOS（C/C++ 实现, 这是一个成熟、工业级的几何库）, python 只是封装了一层，所以速度会比纯 python 快很多


## base


`点、线、面`


```python
from shapely.geometry import Point, LineString, Polygon

point = Point(0, 0)  # 创建点(0,0)
print(point.x, point.y)  # 获取坐标

line = LineString([(0, 0), (1, 1), (2, 0)])  # 创建线
print(line.length)  # 计算长度

polygon = Polygon([(0, 0), (1, 1), (1, 0)])  # 创建多边形
print(polygon.area)  # 计算面积
```


`几何集合`

```python
# 缓冲区分析
buffer = point.buffer(1.0)  # 创建半径为1的缓冲区

# 联合操作
union = polygon.union(other_polygon)

# 差集操作
difference = polygon.difference(other_polygon)

# 交集操作
intersection = polygon.intersection(other_polygon)
```


> 缓冲区指围绕几何对象一定范围内的区域，用于影响范围分析（化工厂污染范围，安全区判定），邻近度分析（查找学校300m 内小区）；创建缓冲区再查找比直接计算距离更高效


`几何属性`

```python
# 获取几何对象的边界
boundary = polygon.boundary

# 获取最小外接矩形
envelope = polygon.envelope


# 获取几何对象的边界
boundary = polygon.boundary

# 获取几何对象的中心点
centroid = polygon.centroid

# 获取最小外接矩形
envelope = polygon.envelope

poly.area        # 面积
poly.length      # 周长
poly.bounds      # 外接矩形边界 (minx, miny, maxx, maxy)
poly.centroid    # 中心点
```


`空间关系`

```python
p.within(poly)         # 点是否在面内
poly.contains(p)       # 面是否包含点
line.intersects(poly)  # 线是否与面相交
poly.equals(other_poly) # 两个面是否相等
```


## demo

1. 可视化展示面积

```python
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# 创建多边形
polygon = Polygon([(0, 0), (1, 1), (1, 0)])
area = polygon.area  # 计算面积

# 可视化
fig, ax = plt.subplots()
x, y = polygon.exterior.xy
ax.fill(x, y, alpha=0.5, fc='blue', ec='black')  # 填充多边形

# 添加面积标签
ax.text(0.5, 0.5, f'面积: {area:.2f}', 
        ha='center', va='center', fontsize=12)

ax.set_aspect('equal')
plt.title('多边形面积可视化')
plt.show()
```









