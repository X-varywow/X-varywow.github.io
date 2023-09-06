
```bash
pip install opencv-python
```


## 1.读取图像&显示图像

demo1: 读取图像&显示图像

- cv2.imread()
- cv2.imshow()
- cv2.imwrite()
- cv2.COLOR_BGR2RGB



!> OpenCV 默认将图像读取为 BGR

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./assets/img1.png')
print(type(img))    # <class 'numpy.ndarray'>
print(img.shape)    # (403, 417, 3) 高 x 宽 x 通道数

# 可以只取图像的部分区域
img = img[height//3:height*2//3, :]
```

```python
# BGR格式，经过转换后才能显示正常
img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imshow("title", img_convert)
plt.imshow(img_convert)


cv2.imwrite("a.jpg", img_convert)
```


demo2: 绘制图形&文字

- cv2.rectangle()
- cv2.circle()
- cv2.putText()



```python
# Rectangle
color=(240,150,240) # Color of the rectangle
cv2.rectangle(img, (100,100),(300,300),color,thickness=10, lineType=8) ## For filled rectangle, use thickness = -1
## (100,100) are (x,y) coordinates for the top left point of the rectangle and (300, 300) are (x,y) coordinates for the bottom right point

# Circle
color=(150,260,50)
cv2.circle(img, (650,350),100, color,thickness=10) ## For filled circle, use thickness = -1
## (250, 250) are (x,y) coordinates for the center of the circle and 100 is the radius

# Text
color=(50,200,100)
font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(img, 'Save Tigers',(200,150), font, 5, color,thickness=5, lineType=20)

# Converting BGR to RGB
img_convert=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_convert)
```


## 2.视频处理

demon3：将视频分帧

- cv2.VideoCapture()
- cap.isOpened()
- cap.read()
- cv2.waitKey()


```python
import cv2

def video_split(v_name, # 读取视频名字
                pic_num, # 视频中帧总数
                framerate # 帧数截取间隔，每隔n帧截取一阵
                ):
 
    v_path = video_path + '\\' + v_name # 读取视频地址
    cap = cv2.VideoCapture(v_path) # 读取视频
 
    while cap.isOpened():
        ret, frame = cap.read()
        # cap.read()表示按帧读取视频。ret和frame是获取cap.read()方法的两个返回值
        # 其中，ret是布尔值。如果读取正确，则返回TRUE；如果文件读取到视频最后一帧的下一帧，则返回False
        # frame就是每一帧的图像

        if success:
            if (pic_num % framerate == 0): # 符合第n帧取帧数图的条件
                cv2.imwrite(save_path + '\\' + str(int(pic_num/framerate)) + '.jpg', frame)
            pic_num += 1
            # cv2.waitKey(1)

        if cv2.waitKey(30) == ord('q'):
            break

    return pic_num # 返回第n个视频的帧总数

```


`k = cv2.waitKey(30) & 0xFF`
- waitKey(30) 会等待30ms, 并返回 ASCII 码 （这是一个 **阻塞函数**，会暂停程序的执行，直到用户按下按键或超过指定时间。在等待期间，是一个轮询来检查键盘状态。轮询 polling ）
- &0xFF 表示按位与，只保留低8位


demo4: 合成视频

```python
capture = cv2.VideoCapture(0)

# 定义编码方式并创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))

while(capture.isOpened()):
    ret, frame = capture.read()

    if ret:
        outfile.write(frame)  # 写入文件
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
```



demo5: 视频常用操作

```python
video = cv2.VideoCapture("video2.mp4")

# 参数为 0 时表示 摄像头
cam = video = cv2.VideoCapture(0)

cv2.CAP_PROP_FPS            # 5

video.get(cv2.CAP_PROP_FPS) # fps 30
```

```python
resized_image = cv2.resize(img, (650,500)) 
```

## 3.操作图像

demo6: 图像数据格式、属性、像素点位的操作

```python
import cv2

img = cv2.imread("hand.png")

img.shape # (480, 320, 3)
height, width, channels = img.shape

img.size # 总像素点数

img.dtype # dtype('uint8')


type(img) # numpy.ndarray

px = img[100, 90]
print(px)

# 默认 GBR 格式
# 只获取蓝色 blue 通道的值
px_blue = img[100, 90, 0]
print(px_blue)
```

demo7:

```python
# 截取固定区域
face = img[100:200, 115:188]
cv2.imshow('face', face)

# 通道分割、合并
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

# 使用索引提效
b = img[:, :, 0]
cv2.imshow('blue', b)
```

demo8: 特定颜色物体追踪

```python
import numpy as np

capture = cv2.VideoCapture(0)

# 蓝色的范围，不同光照条件下不一样，可灵活调整
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])

while(True):
    # 1.捕获视频中的一帧
    ret, frame = capture.read()

    # 2.从 BGR 转换到 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3.inRange()：介于 lower/upper 之间的为白色，其余黑色
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4.只保留原图中的蓝色部分
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(1) == ord('q'):
        break
```


demo9: 图像混合


- cv2.add()
- cv2.addWeighted()
- cv2.bitwise_and(), cv2.bitwise_or(), cv2.bitwise_not()
- cv2.bitwise_xor()

> 掩膜（mask）: 一块挡板，如喷漆中负责遮住区域或留出区域

参考：https://codec.wang/docs/opencv/basic/image-blending


```python
# 图像相加，并附带权重，最后的一个值是一个修正值，为 0
img1 = cv2.imread('lena_small.jpg')
img2 = cv2.imread('opencv-logo-white.png')
res = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
```

```python
img1 = cv2.imread('lena.jpg')
img2 = cv2.imread('opencv-logo-white.png')

# 把 logo 放在左上角，所以我们只关心这一块区域
rows, cols = img2.shape[:2]
roi = img1[:rows, :cols]

# 创建掩膜
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# 保留除 logo 外的背景
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
dst = cv2.add(img1_bg, img2)  # 进行融合
img1[:rows, :cols] = dst  # 融合后放在原图上
```


demo10: 加速示例

```python
# 加速前
for i in range(cartoon_face.shape[0]):
    for j in range(cartoon_face.shape[1]):
        if np.all(cartoon_face[i, j] != [225, 255, 225]):
            frame[i+x_offset, j+y_offset] = cartoon_face[i, j]


# 利用向量化操作加速代码
mask = np.all(cartoon_face != [225, 255, 225])  # , axis=2)
frame[x_offset:x_offset+cartoon_face.shape[0], y_offset:y_offset+cartoon_face.shape[1]] = \
    np.where(mask[..., None], cartoon_face, frame[x_offset:x_offset+cartoon_face.shape[0], y_offset:y_offset+cartoon_face.shape[1]])
```


## 4.几何变换

- cv2.resize()
- cv2.flip()
- cv2.warpAffine()


[图像几何变换](https://codec.wang/docs/opencv/start/image-geometric-transformation)

[番外篇：仿射变换与透视变换](https://codec.wang/docs/opencv/start/extra-05-warpaffine-warpperspective)




## 5.滤波

[番外篇：卷积基础 - 图片边框](https://codec.wang/docs/opencv/basic/extra-08-padding-and-convolution)

[平滑图像](https://codec.wang/docs/opencv/basic/smoothing-images)


> 滤波和模糊，都属于卷积，但卷积核不同；**卷积：扩展到矩阵的点积** </br></br>
> 低通滤波器（去除图像中的高频细节）（如：均值滤波、高斯滤波、中值滤波） 是 模糊， 高通滤波器 是 锐化


demo 11: 各种滤波

- cv2.blur()
- cv2.boxFilter()
- cv2.GaussianBlur()
- cv2.medianBlur()
- cv2.bilateralFilter


`均值滤波`

$$kernel = \frac19 \begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}$$

```python
img = cv2.imread("img.jpg")
blur = cv2.blur(img, (3,3))
```

`方框滤波`

$$kernel = a \begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}$$

```python
# normalize=True 时为 均值滤波
# normalize=True 时， a = 1
blur = cv2.boxFilter(img, -1, (3, 3), normalize=True)
```

`高斯滤波`

类似高斯分布，越靠近中心的像素权重越高。

能有效消除高斯噪声，保留更多的图像细节

```python
gaussian = cv2.GaussianBlur(img, (5, 5), 1)
```

`中值滤波`

取位于中位的像素替代本像素，非线性操作，比上面的滤波方式要慢

```python
median = cv2.medianBlur(img, 5)
```

`双边滤波`

利好边缘信息

```python
blur = cv2.bilateralFilter(img, 9, 75, 75)
```



## 6.边缘检测

[番外篇：图像梯度](https://codec.wang/docs/opencv/basic/extra-09-image-gradients)

[边缘检测](https://codec.wang/docs/opencv/basic/edge-detection)

[阈值分割](https://codec.wang/docs/opencv/start/image-thresholding)


> 低通滤波，只保留图像低频部分，可以平滑模糊图片。</br></br>
> 同理，想要得到物体的边缘，只需要 高通滤波


垂直边缘提取卷积核：
$$kernel = \begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}$$


demo12: 提取边缘信息

- cv2.filter2D()
- cv2.threshold()
- cv2.Canny()

```python
img = cv2.imread('sudoku.jpg', 0)

kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype = np.float32)

# 卷积，-1 表示通道数与原图相同
dst_v = cv2.filter2D(img, -1, kernel)
dst_h = cv2.filter2D(img, -1, kernel.T)

# 横向并排显示
cv2.imshow('edge', np.hstack((img, dst_v, dst_h)))
```



```python
import cv2
import numpy as np

img = cv2.imread('handwriting.jpg', 0)

# 先进行阈值分割会有更好的效果
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# canny 边缘检测
edges = cv2.Canny(thresh, 30, 70)

cv2.imshow('canny', np.hstack((img, thresh, edges)))
cv2.waitKey(0)
```


demo13: 边缘检测，用于形状判断

```python
# 读取图像并进行预处理
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blur, 50, 150)

# 对轮廓形状判断
image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) == 3:
        # 三角形
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    elif len(approx) == 4:
        # 矩形或正方形
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            # 正方形
            cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
        else:
            # 矩形
            cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)
    elif len(approx) == 5:
        # 五边形
        cv2.drawContours(image, [approx], 0, (255, 255, 0), 2)
```

或者：形状识别算法（如Hu矩、Zernike矩等）

## 7.模版匹配

- cv2.matchTemplate()
- cv2.minMaxLoc()

demo14: 模版匹配

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', 0)
template = cv2.imread('face.jpg', 0)
h, w = template.shape[:2]  # rows->h, cols->w

# 相关系数匹配方法：cv2.TM_CCOEFF
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

left_top = max_loc  # 左上角
right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
```

匹配多个物体：

```python
# 1.读入原图和模板
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

# 2.标准相关模板匹配
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

# 3.这边是 Python/Numpy 的知识，后面解释
loc = np.where(res >= threshold)  # 匹配程度大于%80 的坐标 y,x
for pt in zip(*loc[::-1]):  # *号表示可选参数
    right_bottom = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
```




## other

形态学操作：腐蚀、膨胀

霍夫变换用来提取图像中的直线和圆等几何形状

[综合应用：车道检测](https://codec.wang/docs/opencv/basic/challenge-03-lane-road-detection)

还有更多功能，光流、相机等，参考：https://opencv-python-tutorials.readthedocs.io/zh/latest/



对 opencv 封装：[python-imutils包简介使用](https://blog.csdn.net/qq_38463737/article/details/118466096)





--------------

参考资料：
- [opencv 教程1](https://codec.wang/docs/opencv/) ⭐️
- [opencv 教程2](https://opencv-python-tutorials.readthedocs.io/zh/latest/)
- chatgpt
- https://zhuanlan.zhihu.com/p/453759410
- https://cloud.tencent.com/developer/article/1739993
- https://blog.csdn.net/m0_62955589/article/details/123999071