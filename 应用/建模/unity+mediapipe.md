
https://github.com/mmmmmm44/VTuber-Python-Unity

发布两年前，代码质量还正常。[视频](https://www.youtube.com/watch?v=3pBc9Wkzzos)

【支持】live2d 和 unityChan 3D

关键点：mediapipe特征点位，运动学计算，配置项驱动模型

这套方案：查看配置项方便多了，不像 kalidokit ，都不知道配的什么


https://www.youtube.com/watch?v=NYuAU4QUjB8

https://codepen.io/mediapipe/details/KKgVaPJ

https://docs.warudo.app/warudo/v/en/mocap/mediapipe

https://github.com/RimoChan/Vtuber_Tutorial


## python 端

文件结构：
- main.py
- facial_landmark.py
- pose_estimator.py
- stabilizer.py
- facial_features.py
- model.txt
  - get_full_model_points 会在相机姿态估计时使用到


### socket

> 利用 TCP socket 建立本地服务，unity 作为服务端与 python 通信


```python
from facial_landmark import FaceMeshDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
from facial_features import FacialFeatures, Eyes
import sys

port = 5066

def init_TCP():
    port = args.port
    address = ('127.0.0.1', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STRAM)
        s.connect(address)
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print(f"Error while connecting: {e}")
        sys.exit()

def send_info_to_unity(s, args):
    msg = ''
```


### opencv


从 main.py 入口去看，

1. 先建立 cv2.VideoCapture(cam)，默认从 0 号摄像头捕获
2. 创建 FaceMeshDetector()， PoseEstimator， Stabilizer
3. 检测到 faces 更新点位 image_points， iris_image_points
4. 3d点位 转到 2d点位，通过solvePnP 来估计相机的姿态，即相机的旋转矩阵和平移向量。
5. FacialFeatures计算出特征，参考如下表格
6. 



solve_pose_by_all_points 计算 (rotation_vector, translation_vector)






### mediapipe

```python
import mediapipe as mp

# facial
model = mp.solutions.face_mesh.FaceMesh( 
    self.static_image_mode,
    self.max_num_faces,
    True,
    self.min_detection_confidence,
    self.min_tracking_confidence
)

res = model.process(img)

for lanmarks in res.multi_face_landmarks:
    for i, lmk in enumerate(landmarks):
        x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)

# features 手动算的，但已经被 mediapipe 集成了
# 检测的 eye, mouth, 计算类似 (a+b）/c

# kalman 滤波？

# live2d 参数配置项这些：
roll, pitch, yaw, ear_left, ear_right, 
x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right, mar, 
mouth_distance
```







### 处理特征

**没有点位图还是不太好看，只能知道大致流程**

从 468 开始的 10 个点位是，瞳孔点位

**对瞳孔的处理：**

（1）先构建了 iris_image_points 只取 x, y

（2）手写地检测 x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.LEFT)

（3）detect_iris 取（眼睛、瞳孔）点位，向量计算出 x_rate, y_rate， 0-1， 0 表示 left or top



**eye_right**

描述的是眼睛的闭合程度，

处理得很不合理，拿 brow-eye来做

FacialFeatures.eye_aspect_ratio














## unity 端



### 文件结构

- adjust grid layout
- file manager
- hiyori controller 接受列表信息，并包含了更新方式
- hiyori pref 定义了预制体动作的一些上下限，
- save data manager
- tcp server
- ui system 配置按钮的 ui
- value text






### 接收信息 & 配置项

```cs
// TCPServer.cs
server = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
receiveThread = new Thread(new ThreadStart(ReceiveData));

client = server.AcceptTcpClient();
NetworkStream stream = client.GetStream();


while ((length = stream.Read(bytes, 0, bytes.Length)) != 0) {
    var incommingData = new byte[length];
    Array.Copy(bytes, 0, incommingData, 0, length);
    string clientMessage = Encoding.ASCII.GetString(incommingData);

    // call Hiyori Controller to update values
    hiyoriController.parseMessage(clientMessage);
    print("Received message: " + clientMessage);
}
```

信息样例：
0.6379 -90.0000 -14.3335 0.3193 0.3004 0.4928 0.5366 0.3444 0.6200 0.0296 125.1439



11 个数值，依次是：
- roll, pitch, yaw
- eye_left, eye_right （已修改，原本是 ear）
- x_ratio_left,y_ratio_left, x_ratio_right, y_ratio_right （都是描述瞳孔，已修改）
- mar, mouth_dist （闭合率，嘴巴宽度）


### 如何更新

拿到 11 个变量信息，作为单独变量保存

model = this.FindCubismModel(); 这是 live2d 模型

在 LateUpdate() 中更新，

通过 model.Parameters[0].Value 直接更新即可；需要熟悉：model.Parameters，

> 可通过模型的 `cdi.json` 查看 Parameters， 是一个列表，含有：Id， GroupId，Name 组成的字典


| 特征（按序）        | 对应  | 说明 |
| ------------------- | ----- | ---- |
| Angle X             | raw   | 0    |
| Angle Y             | pitch | 1    |
| Angle Z             | roll  | 2    |
| Flush               | √     | 3    |
| Eye L Open          | √     | 4    |
| Eye L Smile         |       | 5    |
| Eye R Open          | √     | 6    |
| Eye R Smile         |       | 7    |
| Eyeball X           | √     | 8    |
| Eyeball Y           | √     | 9    |
| Brow L Form         |       | 10   |
| Brow R Form         |       | 11   |
| Mouth Form          |       | 12   |
| Mouth Open          |       | 13   |
| Body X              |       | 14   |
| Body Y              |       | 15   |
| Body Z              |       | 16   |
| Breath              |       | 17   |
| Arm L A             | √     | 18   |
| Arm R A             | √     | 19   |
| Bust Bounce         |       | 20   |
| Hair Move Ahoge     |       | 21   |
| Hair Move Front     |       | 22   |
| Hair Move Side      | √     | 23   |
| Hair Move Back      |       | 24   |
| Move Side Up        |       | 25   |
| Move Butterfly Tie  |       | 26   |
| Move Skirt          |       | 27   |
| Move Side Up Ribbon |       | 28   |










### 场景信息

live2d 场景：
- Main Camera
- hiyori 模型预制体
  - 一堆live2d 内置的 .cs
  - hiyori controller
- Canvas 按钮文本控件
  - 添加控件
  - setting/tcp panel 附加了 网格布局.cs
- App Controller (两个按钮的设置信息都在这，挂载两个脚本，onclick 提供给 ui 节点)
- EventSystem (standalone input module)













### 代码信息

unity 中也是 5066

```cs
// unity 
listener = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
//监听，传回来的 11 配置项

// for Live2D model
using Live2D.Cubism.Core;
using Live2D.Cubism.Framework;
model = this.FindCubismModel();

// eg. eyeblink
ear_right = Mathf.Clamp(ear_right, ear_min_threshold, ear_max_threshold) // 限制最大最小值

model.Parameters[6].Value = eye_L_value

// 在 LateUpdate 中更新
```

> live2d offical model update:

```cs
using UnityEngine;
using Live2D.Cubism.Core;
using Live2D.Cubism.Framework;

public class ParameterLateUpdate : MonoBehaviour
{
    private CubismModel _model;
    private float _t;

    private void Start()
    {
        _model = this.FindCubismModel();
    }

    private void LateUpdate()
    {
        _t += (Time.deltaTime * 4f);
        var value = Mathf.Sin(_t) * 30f;

        var parameter = _model.Parameters[2];

        parameter.Value = value;
    }
}
```




## todo

当前目标：**hiyori 较好的动捕，视频素材正确处理，只涉及头部**

可以读取json, 通过 Unity Video Recorder 在 update 中捕获并生成动画

都是 unity recorder Window>General>Recorder>Recorder Window


并且这个不涉及面部遮挡，可以的，

最终：开发一个 unity 软件


不如 live2d 直接导出动画，只要负责去拟合就行。


- [ ] python 传递点位
- [ ] unity live2d 生成动画
- [ ] python 拼接图像


- [ ] unity 读取json 并导出动画


- [ ] 那个曲线还没太明白


但是我网上找到的，都是利用摄像头实时渲染，**没有利用视频，然后导出动画的**。。。



`0814`

上午
- [x] unity 文字游戏，学一下控件2 UI 使用
- [x] 修改 mainv2, 直接跳过了 json 这一步，可以的 （动作幅度太小，）
- [x] 整理项目 csdn-build2dperson
下午
- [x] 了解 unity 代码, 控件运行方式
- [x] mediapipe unity plugin (有些复杂，目前动不了，)
- [x] 了解配置项，修改代码，使 unity live2d 更好拟合视频 
- [ ] (3个角度配置项，)
- [ ] unity recorder 教程
- [ ] sence Koharu ，尝试使用 videorecorder, 导出动画
- [ ] 修改 unity 中区域可选方式 （Drawables ArtMesh）
- [ ] 重写 unity 部分，左右两块进行显示
- [ ] 重写代码，作者不知道改的啥，换成视频后很奇怪
晚上
- [ ] 查看更多项目，寻找灵感
- [ ] https://github.com/huihut/Facemoji
- [ ] https://github.com/search?q=unity+live2d&type=repositories&s=stars&o=desc
- [ ] 小狐狸项目

live2d viewer 没有导出视频（功能）





模本版本导致配置项可能对不上


如何调试：
- [x] 打印特征，特征格式
- [x] 摄像头捕捉改为固定图片
- 修改代码，使用最新方式，新增特征；目前不做
- [x] 添加暂停代码，更容易调试，更多信息，等等

- [x] 使用 data.json 保存信息，实时的对于视频并不好。所以测试时，用离线文件来驱动 unity 做动画
- [ ] 改进眼睛计算方式（坑，原本是两端混合运算）： python 端换方式，unity 端都错的（设的阈值是中间计算阈值，而不是结果阈值，）
- [ ] 查看 unity 使用配置项的方式（文件好多。。。。）
- [ ] 查看 model 解
- [ ] unity 动画如何导出
- [ ] 如何将 json 作为 unity 输入

- [ ] 第二个参数 pitch 一直是 -90, 有空看下

- [ ] 改进 python 使用 mediapipe 的方式
- [ ] 使用 mediapipe for unity 


> live2d 还是有些不熟，了解 live 2d，特定模型的 model.Parameters


修改都在 video.py 中
- 变量名，舒服多了
- 添加暂停
- 打印信息
- 修改输入



改进点：
- [ ] 眼睛闭合判得太重了，
- [ ] 使用的手写特征判断，官方的好些
- [ ] 使用 mediaforpipe
- [ ] 还是存在抖动，人脸识别的出来就是具有微小抖动，后期估计有个平滑的大工作，插帧？


- [ ] 更快的通讯方式，共享内存？
- [ ] 应该有个远近检测，为了嵌入原视频的话


## 关键计算 🔍

> 首先，熟悉 mediapipe


### _cv2.solvePnP_

用于通过给定的3D点和对应的2D点来估计相机的姿态（旋转矩阵和平移向量）

使用方法如下：
1. 首先，准备好3D点的坐标（世界坐标系）和对应的2D点的像素坐标（图像坐标系）。
2. 根据所选的算法，选择相机的内参矩阵和畸变系数。
3. 调用cv2.solvePnP函数，传入3D点、2D点、相机内参矩阵和畸变系数，以及所选择的算法。
4. 函数将返回相机的旋转矩阵和平移向量。


示例代码：
```python
import cv2
import numpy as np

# 3D点的坐标
object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
# 2D点的像素坐标
image_points = np.array([[10, 20], [50, 60], [15, 25], [55, 65]], dtype=np.float32)
# 相机内参矩阵
camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
# 畸变系数
dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)

# 调用solvePnP函数
retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

print("旋转矩阵：")
print(cv2.Rodrigues(rvec)[0])
print("平移向量：")
print(tvec)
```

应用场景：

1. 相机姿态估计：通过给定的3D点和对应的2D点，可以使用solvePnP函数来**估计相机的姿态，即相机的旋转矩阵和平移向量**。这在计算机视觉中常用于**目标跟踪、姿态估计和增强现实**等任务。

2. 三维重建：通过将多个图像中的2D点和对应的3D点输入solvePnP函数，可以估计相机的姿态，并进一步进行三维重建。这在计算机视觉中常用于建立三维模型、虚拟现实和室内导航等应用。

3. 视觉里程计：通过连续的图像帧和对应的2D点，可以使用solvePnP函数来估计相机的运动，即相机的旋转矩阵和平移向量。这在机器人领域中常用于视觉里程计、自主导航和SLAM（同时定位与地图构建）等任务。

4. 姿态估计：通过给定的3D模型和对应的2D点，可以使用solvePnP函数来估计物体的姿态，即物体的旋转矩阵和平移向量。这在计算机视觉中常用于物体识别、姿态估计和姿态跟踪等任务。

总之，cv2.solvePnP函数在计算机视觉和机器人领域中用于估计相机或物体的姿态，从而实现目标跟踪、三维重建、视觉里程计和姿态估计等应用



### _kalman_

> 卡尔曼滤波，是一种基于状态估计的数学算法。通过对系统的观测和控制输入进行融合，从而提供对系统当前状态的最优估计。

Kalman filter的原理基于贝叶斯概率推断，它将状态估计问题建模为一个动态系统，该系统包括状态方程和观测方程。状态方程描述了系统的状态如何根据先前的状态和控制输入进行演化，而观测方程描述了如何根据观测值对系统的状态进行测量。

Kalman filter的运行过程可以分为两个步骤：预测和更新。在预测步骤中，基于先前的状态估计和控制输入，通过状态方程预测系统的当前状态。在更新步骤中，利用观测方程将预测的状态与实际观测进行比较，从而纠正预测的误差，并得到最优的状态估计。

Kalman filter的优点在于其能够处理包含噪声和不确定性的系统，并且能够自适应地调整权重来平衡先验信息和观测信息的贡献。它在许多领域中广泛应用，例如导航系统、机器人技术、信号处理等。

总的来说，Kalman filter是一种用于状态估计的优化算法，能够通过融合观测和控制输入，提供对系统当前状态的最优估计。它的应用范围广泛，并且具有较好的鲁棒性和自适应性。






--------------


kalman filter在 Stabilizer 中， 使用方式：

```python
mouth_dist_stabilizer = Stabilizer(...)
eyes_stabilizers = [Stabilizer(...) for _ in range(6)]

mouth_dist_stabilizer.update([mouth_dist])
steady_mouth_dist = mouth_dist_stabilizer.state[0]

for v, stb in eyes_stabilizers:
    stb.update([v])
    steady_pose_eye.append(stb.state[0])

# 就是每个点位有一个单独的 稳定器，将新的点位放去 update ，产生的结果作为最终点位
```


| 参数         | 说明 |
| ------------ | ---- |
| state_name   | 2    |
| measure_name | 1    |
| cov_process  | 0.1  |
| cov_measure  | 0.1  |


!> 运行一下 stabilizer.py 报错， 将 array 类型使用 int(state[0]) 改成 int 型；这个测试程序功能：跟踪鼠标画点，然后预测？

27 是 Esc 记一下，以后 opencv 经常会用。


----------------



卡尔曼滤波器是一个递归的状态估计滤波器，可以用于估计状态变量的值。下面是一个使用Python实现卡尔曼滤波器的示例代码：

```python
import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        self.A = A  # 状态转移矩阵
        self.B = B  # 控制输入矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 状态噪声协方差
        self.R = R  # 观测噪声协方差
        self.P = P  # 估计误差协方差
        self.x = x  # 状态估计向量

    def predict(self, u=0):
        # 预测步骤
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x
    
    def update(self, z):
        # 更新步骤
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)
        return self.x

# 示例用法
A = np.array([[1, 1], [0, 1]])  # 状态转移矩阵
B = np.array([[0.5], [1]])  # 控制输入矩阵
H = np.array([[1, 0]])  # 观测矩阵
Q = np.array([[0.0001, 0], [0, 0.0001]])  # 状态噪声协方差
R = np.array([[1]])  # 观测噪声协方差
P = np.array([[1, 0], [0, 1]])  # 估计误差协方差
x = np.array([[0], [0]])  # 初始状态估计向量

kf = KalmanFilter(A, B, H, Q, R, P, x)

# 预测和更新
prediction = kf.predict()
print("预测值：", prediction)

measurement = np.array([[1.2]])
update = kf.update(measurement)
print("更新值：", update)
```

在上述代码中，我们定义了一个`KalmanFilter`类，构造函数初始化了卡尔曼滤波器的参数。`predict`方法用于预测下一个状态值，`update`方法用于根据观测值更新状态估计值。通过调用`predict`和`update`方法可以实现卡尔曼滤波器的预测和更新过程。

此外，我们还定义了一个简单的示例用法，其中使用了一个二维状态向量和一个一维观测向量来进行状态估计。










--------------



参考资料：
- chatgpt
- [live2d-关于模型参数更新](https://docs.live2d.com/zh-CHS/cubism-sdk-tutorials/about-parameterupdating-of-model/)