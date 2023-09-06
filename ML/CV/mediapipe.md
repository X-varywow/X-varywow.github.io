


[google developer 文档](https://developers.google.com/mediapipe/solutions/guide) 

github地址：https://github.com/google/mediapipe

【介绍】视频和图片的机器学习解决方案

【功能】脸部检测、手势检测、姿态检测等。

【支持】web, python, Android

```bash
pip install mediapipe
```



## Face Detection

参考：[colab 代码](https://colab.research.google.com/drive/1JaQL7MnsH1NN6i5wsloWlIYhoUC4QZOh)


```python
import mediapipe as mp
import matplotlib.pyplot as plt
import math


mp_face_detection = mp.solutions.face_detection

# help(mp_face_detection.FaceDetection)

short_range_images = {name: cv2.imread(name) for name in ['person_head.png']}

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    # cv2_imshow(img)
    # cv2_imshow 有可能会报错
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换格式以正常显示，不然蓝蓝的

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 通过 model_selection=1 可以指定 full range model
# model_selection=0 适用于 大头照
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5,
    model_selection=0) as face_detection:
    for name, image in short_range_images.items():
        res = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print(f"Face detections of {name}:")
        if not res.detections:
            continue
        annotated_image = image.copy()
        for detection in res.detections:
            mp_drawing.draw_detection(annotated_image, detection)
        resize_and_show(annotated_image)
```


## Face landmarks 🔍

[官方DEMO演示](https://mediapipe-studio.webapps.google.com/studio/demo/face_landmarker)

> 功能：实时计算 **478个面部点位** 和 **52个面部特征**。可用于虚拟主播、视频动捕等。

计算的是3D 空间的点位信息，包含 z 轴方向信息

包含了3个模型 [下载地址](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)


| 模型         | 输入        | 结构                                                                                              | 说明         |
| ------------ | ----------- | ------------------------------------------------------------------------------------------------- | ------------ |
| FaceDetector | 192 x 192   | Convolutional Neural Network: [SSD](https://arxiv.org/abs/1512.02325)-like with a custom encoder. | 都是 float16 |
| FaceMesh-V2  | 256 x 256   | Convolutional Neural Network: [MobileNetV2](https://arxiv.org/abs/1801.04381)                     |              |
| Blendshape   | 1 x 146 x 2 | [MLP-Mixer](https://keras.io/examples/vision/mlp_image_classification/#the-mlpmixer-model)        |              |



other: 使用 Effect Renderer 可以添加特效贴图等等，

### 示例1

参考：[colab 代码](https://colab.research.google.com/drive/1FCxIsJS9i58uAsgsLFqDwFmiPO14Z2Hd)

```python
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
help(mp_face_mesh.FaceMesh)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

images = {name: cv2.imread(name) for name in ['person_head.png']}

with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    refine_landmarks = True,
    max_num_faces = 2,
    min_detection_confidence = 0.5) as face_mesh:

    for name, image in images.items():
        res = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print(f"Face landmarks of {name}:")
        if not res.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in res.multi_face_landmarks:

            # 画出灰色的脸部网格
            mp_drawing.draw_landmarks(
                image = annotated_image,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        resize_and_show(annotated_image)
```


添加其他的绘制：

```python
# 绘制轮廓，包括：眼睛、脸部、眉毛、嘴巴
mp_drawing.draw_landmarks(
    image = annotated_image,
    landmark_list = face_landmarks,
    connections = mp_face_mesh.FACEMESH_CONTOURS,
    landmark_drawing_spec = None,
    connection_drawing_spec = mp.drawing_styles.get_default_face_mesh_contours_style()
)

# 绘制虹膜（四边形）
mp_drawing.draw_landmarks(
    image=annotated_image,
    landmark_list=face_landmarks,
    connections=mp_face_mesh.FACEMESH_IRISES,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
)
```








### 示例2

参考: [github face_mesh](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md)

基本流程：

```python
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh() as face_mesh:
    res = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for face_landmarks in res.multi_face_landmarks:
        pass
```

利用 CV2 捕获视频：

```python
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGR)
        results = face_mesh.process(image)
        
        
        
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
```






### 示例3

参考: [官方代码](https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb)

不用 mp.solutions 了 用 mp.tasks

**3个重要功能：**
- 面部网格
- 形状特征
- transformation matrix


```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(rgb_image, res):
    annotated_image = rgb_image.copy()
    for face_landmarks in res.face_landmarks:
        
        # 做一次点位转换，作用？
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in face_landmarks
        ])
        
        mp_drawing.draw_landmarks(
            image = annotated_image,
            landmark_list = face_landmarks_proto,
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image = annotated_image,
            landmark_list = face_landmarks_proto,
            connections = mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image = annotated_image,
            landmark_list = face_landmarks_proto,
            connections = mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image


# plot_face_blendshapes_bar_graph
def show_feature(bs):
    names = [i.category_name for i in bs]
    scores = [i.score for i in bs]
    ids = range(len(names))
    
    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(ids, scores, label = [str(i) for i in ids])
    ax.set_yticks(ids, names)
    ax.invert_yaxis()
    
    for score,patch in zip(scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()
    
    

base_options = python.BaseOptions(model_asset_path = 'face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options = base_options,
                                      output_face_blendshapes = True,
                                      output_facial_transformation_matrixes = True,
                                      num_faces = 1)

detector = vision.FaceLandmarker.create_from_options(options)

# 使用 4 通道的png会报错
image = mp.Image.create_from_file("person2jpg.jpg")
res = detector.detect(image)

# 功能1： 面部网格
annotated_image = draw_landmarks(image.numpy_view(), res)
plt.imshow(annotated_image)

# 功能2：形状特征 (52 个面部特征)
show_feature(res.face_blendshapes[0])

# 功能3：transformation matrix
print(res.facial_transformation_matrixes)
```


使用 通道不符的 png 可能报错，转化一下：

```python
from PIL import Image

img = Image.open("person_head.png")
img = img.convert('RGB') 
img.save("person2jpg.jpg")
```

### 示例4 🔍

使用 mp.tasks，参考 [开发者文档](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python)

需要先下载模型文件

```python
import mediapipe as mp
from pprint import pprint

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,                # 默认为 False
    output_facial_transformation_matrixes=True,  # 默认为 False
    running_mode=VisionRunningMode.IMAGE)


# Load the input image from an image file.
mp_image = mp.Image.create_from_file('image.jpg')

# Load the input image from a numpy array.
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    
with FaceLandmarker.create_from_options(options) as landmarker:
    res = landmarker.detect(mp_image)
    # print(res,type(res))
    # print(res.face_landmarks)
    pprint(res.face_blendshapes)
    # print(res.facial_transformation_matrixes)


# 拿到 res 就不必受限了，直接取就行
# 如 res.face_landmarks[0] ，也不必像示例3 一样 Normalize(提前做了)
```

更多的取法：

```python
res.face_landmarks[0]       #第一个人
res.face_landmarks[0][0]    #第一个人第一个点位
res.face_landmarks[0][0].x

[i.category_name for i in res.face_blendshapes[0]]  # 查看 52 个面部特征，的名字

# res.face_blendshapes[0]: 
#       index 0~51
#       socre 0~1
#       category_name
```






> 文档写得一坨浆糊，mediapipe.tasks 与 mediapipe.solutions 两套代码忍了，官方同一个示例中还整 mediapipe.tasks.python.vision 和 mediapipe.tasks.vision 两套



### 示例5

看看其他 vtuber 的代码是怎么用mediapipe 的





## Hand landmarks

参考：[colab 代码](https://colab.research.google.com/drive/1FvH5eTiZqayZBOHZsFm-i7D-JvoB9DVz)


```python
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# help(mp_hands.Hands)

images = {name: cv2.imread(name) for name in ['hand.png']}

with mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,
    min_detection_confidence = 0.7) as hands:

    for name, image in images.items():
        res = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

        print(f"Handedness of {name}:")
        print(res.multi_handedness)  # 有个分类器，会分出左右手

        if not res.multi_hand_landmarks:
            continue

        print(f"Hand landmarks of {name}")
        image_hight, image_width, _ = image.shape
        annotated_image = cv2.flip(image.copy(), 1)

        for hand_landmarks in res.multi_hand_landmarks:
            # 这样子取点 .landmark[].x，并且mediapipe 是处理成比例点，最大为1， 需要 *image_width
            print(
                f'Index finger tip coordinate ',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width},',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight}'
            )
            mp_drawing.draw_landmarks(
              annotated_image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
        resize_and_show(cv2.flip(annotated_image, 1))
```

?> `cv2.flip(src, flipCode[, dst])` 将图片进行翻转， 0 水平翻转（沿 y 方向），1 垂直翻转，-1 同时翻转


可以在 3d 空间绘制，牛的，从 2d 图片中就可以提出 3d 的信息

```python
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7) as hands:
    
    for name, image in images.items():
        res = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print(f'Hand world landmarks of {name}:')
        if not res.multi_hand_world_landmarks:
            continue

        for hand_world_landmarks in res.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
```







参考：https://cloud.tencent.com/developer/article/2081312


```python
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res = hands.process(imgRGB)

if res.multi_hand_landmarks:
    for handLms in res.multi_hand_landmarks:
        pass

```




## Pose landmark

参考： [官方文档](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) [github 文档](https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md)



```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
```











## Holistic landmarks


> 一个集成的模块，姿态检测 + 手部检测 + 面部检测 + 素材切图

参考：[官方colab](https://colab.research.google.com/drive/16UOYQ9hPM6L5tkq7oQBl1ULJ8xuK5Lae)


```python
import mediapipe as mp
import matplotlib.pyplot as plt


mp_holistic = mp.solutions.holistic
# help(mp_holistic)

mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

images = {name: cv2.imread(name) for name in ["pose.png"]}


with mp_holistic.Holistic(
    static_image_mode = True,
    min_detection_confidence = 0.5,
    model_complexity = 2) as holistic:
    
    for name, image in images.items():
        res = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_high, image_width, _ = image.shape
        
        print(
            f'Nose coordinates: ('
            f'{res.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            f'{res.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
        )

        print(f'Pose landmarks of {name}:')
        
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image,
            res.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            res.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        resize_and_show(annotated_image)
```

同样地，可以在 3D 空间中绘制表示

```python
# Run MediaPipe Holistic and plot 3d pose world landmarks.
with  mp_holistic.Holistic(static_image_mode=True) as holistic:
    for name, image in images.items():
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print the real-world 3D coordinates of nose in meters with the origin at
        # the center between hips.
        print('Nose world landmark:'),
        print(results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])

        # Plot pose world landmarks.
        print(f'Pose world landmarks of {name}:')
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
```

## 论文阅读


[Attention Mesh: High-fidelity Face Mesh Prediction in Real-time](https://arxiv.org/pdf/2006.10962.pdf)

先是经过 Blaze Face Detector 人脸检测模型，之后对特征区域提取局部特征。

注意力机制具体的做法为：在特征空间上设置二维网状采样点，并使用二维高斯核或者仿射变换与微分差值（differen-tiable interpolations）在采样点对应的特征图上提取特特征。



---------------

参考资料：
- https://yinguobing.com/attention-mesh-as-face-mesh-solution-from-google/






## other


MediaPipe Iris 专门用于虹膜，能够计算出深度等（在可变光照、遮挡物等情况）

[应用：交互抠图](https://mediapipe-studio.webapps.google.com/demo/interactive_segmenter)

提供了一个思路：如果后续抠图不够干净的时候，可以采用这个方式细致抠图。

或者把这个部署到网页，或者更换模型




