

> live 2d 可以的，这对以后制作游戏素材，制作动画等，非常有用

[模型下载](https://www.live2d.com/zh-CHS/download/sample-data/)


[视频教程1](https://www.bilibili.com/video/BV1JE411Y7Te/)

https://www.live2d.com/zh-CHS/download/content-guide/

更多教程：https://www.xbeibeix.com/video/BV1WB4y1y7cg

[live2d 动画制作流程](https://blog.csdn.net/wangyiyungw/article/details/83015827)

[live2d 流程教程1](https://www.yuanhuaren.com/course/804)


图层切图 + 视觉变换，已经可以做很多了（真实图层，特效图层，）


添加动态管理器，即关键帧驱动形成的动画。



## 文件说明

- model.json
  - 配置模型、材质、物理、动画文件，groups（就是打包动作起别名）
  - motion
    - idle 会一直循环播放
- physics.json
- .moc3
- cdi.json
  - version
  - parameters (对应 Id GroupId Name, name 在编辑器中使用，滑条调控数值)
  - parameterGroups
  - Parts（Core, Cheek, Brow 等）
  - CombinedParameters
- motion
  - motion.json
    - Version
    - Meta
    - Curves 一个 {} 组成的 []
      - Target "Parameter"
      - Id
      - Segments
- texture



```js
{
  "Target": "Parameter",
  "Name": "LipSync",
  "Ids": [
    "ParamMouthOpenY"
  ]
}
```







## other

[提高动态质量的技巧](https://docs.live2d.com/zh-CHS/cubism-editor-tutorials/motion-hint/)


米哈游，先行展示页面

[你管这玩意儿叫live2D？是live2D 但是米哈游！](https://www.bilibili.com/video/BV13A4y1X75S)


Live2DViewerEX

类似的，搜虚拟主播，motionface vtuberface
