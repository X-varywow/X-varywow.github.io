
## _preface_

支持全方位的视觉 AI 任务，包括目标检测、实例分割、图像分类、姿势估计、跟踪


> （自用）主要任务：游戏中,检测怪物和掉落物，用于后续用途；后续“本任务”都指这个




参考：[官方 colab tutorial](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) ⭐️





```python
%pip install ultralytics
import ultralytics
ultralytics.checks()
```

```python
!yolo predict model=yolo11n.pt source='https://ultralytics.com/images/zidane.jpg'
```



```python
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")
```





</br>

## _流程_


### 格式定义

https://docs.ultralytics.com/datasets/detect/


```python
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8 # dataset root dir (absolute or relative; if relative, it's relative to default datasets_dir)
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
# test: # test images (optional)

# Classes (80 COCO classes)
# 定义 label, 检测目标
names:
    0: person
    1: bicycle
    2: car
    # ...
    77: teddy bear
    78: hair drier
    79: toothbrush
```


标注工具：[LabelImg](https://github.com/HumanSignal/labelImg), 最后使用了 [label studio](https://labelstud.io/guide/quick_start)


标注好了之后按照 yolo 格式导出，然后自己新建一个 data.yaml 即可，注意编号与标注时对应。





### 训练

https://docs.ultralytics.com/modes/train


```python
!yolo train model=yolo11n.pt data=coco8.yaml epochs=3 imgsz=640
```

命令行方式比较便捷，标注好数据，定义好 数据描述文件 `.yaml` 就差不多可以训练了； `n` 表示 nano, 最小的。


-------------

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.yaml')  # build a new model from scratch
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='coco8.yaml', epochs=10)  # train the model
results = model.val()  # evaluate model performance on the validation set
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
results = model.export(format='onnx')  # export the model to ONNX format
```

默认 imgsz=640， 会把图片缩放统一大小，(保持长宽比，增加黑边)


--------------

export 相关，

💡 ProTip: Export to ONNX or OpenVINO for up to 3x CPU speedup.

💡 ProTip: Export to TensorRT for up to 5x GPU speedup.




--------------



1. detection

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # load a pretrained YOLO detection model
model.train(data='coco8.yaml', epochs=3)  # train the model
model('https://ultralytics.com/images/bus.jpg')  # predict on an image
```

2. segmentation

```python
from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')  # load a pretrained YOLO segmentation model
model.train(data='coco8-seg.yaml', epochs=3)  # train the model
model('https://ultralytics.com/images/bus.jpg')  # predict on an image
```

3. classification

```python
from ultralytics import YOLO

model = YOLO('yolo11n-cls.pt')  # load a pretrained YOLO classification model
model.train(data='mnist160', epochs=3)  # train the model
model('https://ultralytics.com/images/bus.jpg')  # predict on an image
```




### 推理


half 半精度推理（float16）

```python
model.predict(source='your_test_images/', half=True)
```

---------------

命令行方式

```python
# Run inference on an image with YOLO11n
!yolo predict model=yolo11n.pt source='https://ultralytics.com/images/zidane.jpg'
```






### 验证

https://docs.ultralytics.com/modes/val

```python
# Download COCO val
import torch
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017val.zip', 'tmp.zip')  # download (780M - 5000 images)
!unzip -q tmp.zip -d datasets && rm tmp.zip  # unzip
```

```python
!yolo val model=yolo11n.pt data=coco8.yaml
```

yolo11n.pt  6 mb...


----------------

指标介绍：

```
YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 1/1 [00:01<00:00,  1.10s/it]
                   all          4         17      0.583       0.85      0.849       0.65
                person          3         10        0.6        0.6      0.591       0.28
                   dog          1          1      0.549          1      0.995      0.796
                 horse          1          2      0.552          1      0.995      0.674
              elephant          1          2      0.369        0.5      0.525       0.26
              umbrella          1          1      0.573          1      0.995      0.995
          potted plant          1          1      0.857          1      0.995      0.895
```

- Box(p), 准确率 $\frac{TP}{TP+FP}$，边界框中平均 58.3% 是判断准确的正例；可能原因：样本少、遮挡等
- Box(r), 召回率 $\frac{TP}{TP+FN}$，检测出了 85% 的真实目标
- mAP50, IoU阈值为0.5时的平均精度
- mAP50-95, IoU阈值从0.5到0.95的平均精度


`IoU`, Intersection over Union, 交并比; 用于衡量预测边界框与真实边界框重叠程度的指标。

$$IoU = \frac{Area\ of\ Overlap}{Area\ of\ Union} = \frac{预测框 \cap 真实框}{预测框 \cup 真实框}$$

通常认为 IoU >= 0.5 时检测有效；

`mPA`, mean Average Precision. 所有类型AP（等同PR 曲线下的面积）的平均值

本任务中，关注 mAP50, 对边界框只需要大致定位即可，掉落物、怪物等...






## _other_


### 目标跟踪

https://docs.ultralytics.com/modes/track

```python
import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
```


### 原理

将目标检测视为回归问题，将图像划分为区域，并预测每个区域的边界框和类别概率







--------------


explorer, 与 llm 结合一起了

https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb


