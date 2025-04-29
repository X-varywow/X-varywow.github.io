
## _preface_

支持全方位的视觉 AI 任务，包括目标检测、实例分割、图像、分类姿势估计、跟踪


> （自用）主要任务：游戏中，用于检测怪物和掉落物，用于后续用途




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
test: # test images (optional)

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





使用工具：LabelImg



### 训练

https://docs.ultralytics.com/modes/train


```python
!yolo train model=yolo11n.pt data=coco8.yaml epochs=3 imgsz=640
```

💡 ProTip: Export to ONNX or OpenVINO for up to 3x CPU speedup.
💡 ProTip: Export to TensorRT for up to 5x GPU speedup.

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.yaml')  # build a new model from scratch
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='coco8.yaml', epochs=3)  # train the model
results = model.val()  # evaluate model performance on the validation set
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
results = model.export(format='onnx')  # export the model to ONNX format
```


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


主要用于检测任务，


https://docs.ultralytics.com/modes/train/#multi-gpu-training





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



## _other_


explorer, 与 llm 结合一起了

https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb


