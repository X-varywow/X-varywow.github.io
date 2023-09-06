
_dlib_

dlib 是一个人脸检测、特征点检测的第三方库，有c++、Python的接口。

dlib 官网： http://dlib.net/

github 地址：https://github.com/davisking/dlib

下载模型文件：http://dlib.net/files/


!> 20230815, 感觉目前 mediapipe 会比这个更好，github 23k star > 12k


</br>

示例1（检测人脸）：

```python
import sys

import dlib

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
if (len(sys.argv[1:]) > 0):
    img = dlib.load_rgb_image(sys.argv[1])
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))

```

</br>

示例2（检测特征点）：


```python
predictor_path = f'shape_predictor_68_face_landmarks.dat' # 68个特征点的检测模型，需下载
predictor = dlib.shape_predictor(predictor_path)

detector = dlib.get_frontal_face_detector()

frame = cv2.imread(frame_path)
face = detector(frame)[0]

landmarks = predictor(frame, face)
points = []
for i in range(68):
    x = landmarks.part(i).x
    y = landmarks.part(i).y
    points.append((x, y))
```