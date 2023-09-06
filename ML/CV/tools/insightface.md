
State-of-the-art 2D and 3D Face Analysis Project

仓库地址：https://github.com/deepinsight/insightface

> 适用于对检测准确度有要求的场景



demo1: 使用 FaceAnalysis

```python
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
```


</br>

demo2: 使用自定义模型

```python
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# detection
detector = insightface.model_zoo.get_model('your_detection_model.onnx')
detector.prepare(ctx_id=0, input_size=(640, 640))

# recognition
handler = insightface.model_zoo.get_model('your_recognition_model.onnx')
handler.prepare(ctx_id=0)
```



---------

参考资料：
- [quick start](https://github.com/deepinsight/insightface/tree/master/python-package)