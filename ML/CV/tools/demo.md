
## (1) 人脸检测&关键点检测

```bash
git clone https://huggingface.co/camenduru/shape_predictor_68_face_landmarks
```
获取模型：http://dlib.net/files/


```python
import cv2
import dlib
import matplotlib.pyplot as plt

# 关键点数量： 5， 68
def check_point(frame_path, cnt = 68):

    # 加载人脸关键点检测器
    predictor_path = f'shape_predictor_{cnt}_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # 加载人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 读取图像&转换灰度
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = detector(frame)
    points = []

    for face in faces:
        # 关键点检测
        landmarks = predictor(frame, face)
        for i in range(cnt):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))
            
    return points
```


## (2) 人脸对齐&仿射变换

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230726212400.jpg">

非共线的三个对应点确认唯一的 **仿射变换**

**透视变换** 需要非共线的四个点唯一确定。（可以用于矫正图片，如"扫描全能王"）


| 变换 | 自由度 | 保持性质                     | 说明                |
| ---- | ------ | ---------------------------- | ------------------- |
| 平移 | 2      | 方向/长度/夹角/平行性/直线性 |                     |
| 刚体 | 3      | 长度/夹角/平行性/直线性      | 旋转+平移           |
| 相似 | 4      | 夹角/平行性/直线性           | 旋转+平移+放缩      |
| 仿射 | 6      | 平行性/直线性                | 旋转+平移+放缩+倾斜 |
| 透视 | 8      | 直线性                       | 模拟人眼近大远小    |



仿射变换：
```python
# 变换前的三个点
pts1 = np.float32([[50, 65], [150, 65], [210, 210]])
# 变换后的三个点
pts2 = np.float32([[50, 100], [150, 65], [100, 250]])

# 生成变换矩阵
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
```


透视变换：
```python
# 原图中卡片的四个角点
pts1 = np.float32([[148, 80], [437, 114], [94, 247], [423, 288]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 0], [320, 0], [0, 178], [320, 178]])

# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换，参数 3 是目标图像大小
dst = cv2.warpPerspective(img, M, (320, 178))
```

利用仿射，简单人脸对齐：
```python
# X 是检测到的5个人脸关键点，Y 是 标准人脸关键点坐标

import numpy as np
from skimage import transform

X, Y = np.array(X), np.array(Y)
tform = transform.SimilarityTransform()

# 程序直接估算出转换矩阵M（SVD分解）
tform.estimate(X, Y)
M = tform.params[0:2, :]
warped = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

plt.imshow(warped)
plt.show()
```

-----------

参考资料：
- https://cloud.tencent.com/developer/article/1894139
- [番外篇：仿射变换与透视变换](https://codec.wang/docs/opencv/start/extra-05-warpaffine-warpperspective)⭐️
- [仿射变换及其变换矩阵的理解](https://www.cnblogs.com/shine-lee/p/10950963.html)

## (2-) dlib 人脸对齐

另一流程，[参考](https://blog.csdn.net/Roaddd/article/details/111866756)

```python
import cv2
import dlib
import matplotlib.pyplot as plt

# 获取图片
my_img = cv2.imread('my_img.jpg')
# BGR to RGB
my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)

# 使用特征提取器get_frontal_face_detector
detector = dlib.get_frontal_face_detector()

dets = detector(my_img, 1)

for det in dets:
    # 将框画在原图上
    # cv2.rectangle  参数1：图片， 参数2：左上角坐标， 参数2：左上角坐标， 参数3：右下角坐标， 参数4：颜色（R,G,B）， 参数5：粗细
    my_img = cv2.rectangle(my_img, (det.left(),det.top()), (det.right(),det.bottom()), (0,255,0), 5)
    
# plt.figure(figsize=(5,5))
plt.imshow(my_img)
plt.show()
```

```python
# 关键点检测
predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')

for det in dets:
    shape = predictor(my_img, det)
    for i in range(68):
        cv2.putText(my_img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(my_img, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255))

plt.imshow(my_img)
plt.show()
```


```python
# 人脸对齐
my_img = dlib.get_face_chip(my_img, shape, size = 150) # size 为输出图形大小
plt.imshow(my_img)
plt.show()
```

[其他人脸对齐文章](https://blog.csdn.net/qq_33221533/article/details/105451558)

https://cloud.tencent.com/developer/article/1394748


## (3) 视频人像添加特效

总体思路：人脸识别 -> 贴图替换


单帧的贴图特效制作：

```python
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

def pic2cartoon(frame, cartoon_image):

    # 加载人脸关键点检测器
    predictor_path = 'shape_predictor_5_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # 人脸检测
    detector = dlib.get_frontal_face_detector()
    faces = detector(frame)
    if not faces:
        print("no face")

    for face in faces:
        landmarks = predictor(frame, face)


        # 获取人脸关键点坐标
        points = []
        for i in range(5):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))

        cartoon_face = cartoon_image.copy()
        cartoon_points = check_point(cartoon_image_path)
        cartoon_points = [cartoon_points[0], cartoon_points[2], cartoon_points[4]]


        src_points = np.array([points[0], points[2], points[4]], dtype=np.float32)
        dst_points = np.array(cartoon_points, dtype=np.float32)  # 目标对齐点
        M = cv2.getAffineTransform(dst_points,src_points)

        # M = cv2.getPerspectiveTransform(np.float32(cartoon_points[:4]), np.float32(points[:4]))
        wraped = cv2.warpAffine(cartoon_face, M, (frame.shape[1], frame.shape[0]))

        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if np.all(wraped[i, j]) and not np.all(wraped[i, j] == 128):
                    # print(wraped[i, j])
                    # 默认填充灰色。。
                    # print(frame[i, j],overlay[i, j])

                    frame[i, j] = wraped[i, j]  # 将贴图像素值复制到帧上
                    
    return frame


cartoon_image_path = '2-removebg.png'
cartoon_image = cv2.imread(cartoon_image_path)

frame_path = 'person_head.png'
frame = cv2.imread(frame_path)

res = pic2cartoon(frame, cartoon_image)
plt.imshow(res)
```



```python
# 感觉速度好慢
video = cv2.VideoCapture("video2.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))


cartoon_image_path = '2-removebg.png'
cartoon_image = cv2.imread(cartoon_image_path)


while True:
    # 读取视频帧
    ret, frame = video.read()
    
    if not ret:
        break
    
    # 将人像贴图特效添加到视频帧上
    res = pic2cartoon(frame, cartoon_image)
    
    # 写入输出视频帧
    output.write(res)
    
video.release()
output.release()
```

## (3-) 改进版

- 简化逻辑，提高效率
- 加入匹配机制（目前只新增嘴巴这一自由度，即根据嘴巴闭合匹配合适的贴图）
- 加入进度条，更加友好
- 加入防闪机制（将前一帧的贴图作为检测不到的默认贴图）

```python
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from math import dist
from tqdm import tqdm


slience = "./slience-removebg-preview.png"
speak = "./speak-removebg-preview.png"


# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.4

def mouth_aspect_ratio(mouth):
    A = dist(mouth[50], mouth[58]) # 51, 59
    B = dist(mouth[52], mouth[56]) # 53, 57
    C = dist(mouth[48], mouth[54]) # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar


def video2cartoon(video_path, cartoon, out_path = 'output.mp4', config = None):
    
    # 定义输入输出
    video = cv2.VideoCapture(video_path)
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # total_frames
    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(video.get(cv2.CAP_PROP_FPS))
    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    
    # 定义检测器
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    
    for i in tqdm(range(n)):
        ret, frame = video.read()
        
        if not ret:
            print("读取视频失败")
        
        if not i:
            pre_frame = frame.copy()
    
        
        faces = detector(frame)
        if not faces:
            print("no face")
        else:
            face = faces[0]
            landmarks = predictor(frame, face)

            points = []
            for pos in landmarks.parts():
                points.append([pos.x, pos.y])


            mar = mouth_aspect_ratio(points)
            if mar > MOUTH_AR_THRESH:
                cartoon_face = cv2.imread(cartoon["open"])
            else:
                cartoon_face = cv2.imread(cartoon["close"])

            cartoon_points = [[100,185],[248,185],[175,240]]


            src_points = np.array([points[36], points[45], points[33]], dtype=np.float32) # 左右，人中
            dst_points = np.array(cartoon_points, dtype=np.float32)  # 目标对齐点

            M = cv2.getAffineTransform(dst_points,src_points)
            pre_frame = cv2.warpAffine(cartoon_face, M, (frame.shape[1], frame.shape[0]))


        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if np.all(pre_frame[i, j]): # 不是黑色（全 0）， 默认这个wraped周围全是黑色
                    frame[i, j] = pre_frame[i, j]  # 将贴图像素值复制到帧上
                    
        output.write(frame)  
        
    video.release()
    output.release()
        

video_path = "video2.mp4"

cartoon = {
    "open": "./speak-removebg-preview.png",
    "close": "./slience-removebg-preview.png"
}

video2cartoon(video_path, cartoon)
```

闭合检测参考：http://www.powersensor.cn/p3_demo/demo8-dlib.html


## (4) 改进

加入 live2d ，更完善的机制

检测模型更换为 google 的 mediapipe, [参考文章](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/)




---------------

更多资料：
- [使用 OpenCV 和 Python 进行人脸对齐](https://bbs.huaweicloud.com/blogs/318085)
- [68 关键点示例](https://raw.githubusercontent.com/vipstone/faceai/master/res/68.jpg)