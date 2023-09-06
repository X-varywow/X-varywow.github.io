
基于项目： unity + mediapipe, 但全部使用的 unity c#

布局：两个等长宽的视频播放， 一个作为输入，一个作为 live2d 动画，可导出。


?> 命名 0814allin, 主要为了 使用 MediaPipeUnityPlugin 开发应用



## _MediaPipeUnityPlugin_

开源地址：[MediaPipeUnityPlugin](https://github.com/homuler/MediaPipeUnityPlugin)


### hello, hand tracking

参考：https://blog.csdn.net/EdmundRF/article/details/130126940


场景结构
- 基本的，Main Camera, Directional Light, EventSystem
- Canvas (设计得像熟悉的网页)
  - body
    - annotatable layer
  - header
    - menu button 挂载 ModalButton.OpenAndPause
  - footer
    - 4 个控制 bu
- Solutions 脚本节点挂载脚本
  - Hand Tracking Solution
  - Hand Tracking Graph
  - Texture Frame pool
  - Bootstrap (页面布局框架？)
  - Web Cam Source
  - Static Image Source
  - Video source


## 流程

mediapipe 中有 ":face_landmarker_v2_with_blendshapes.task"



















--------------

参考资料：
- [Getting Started](https://github.com/homuler/MediaPipeUnityPlugin/wiki/Getting-Started)
- https://blog.csdn.net/EdmundRF/article/details/130126940