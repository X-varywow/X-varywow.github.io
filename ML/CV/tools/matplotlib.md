

在图上绘制散点（人脸关键点检测会用到）(直接 scatter 就行)：

```python
import matplotlib.pyplot as plt

x, y = [i[0] for i in points], [i[1] for i in points]

im = plt.imread("person_head.png")
plt.imshow(im)
plt.scatter(x, y,  c='b', s=10)

plt.show()
```