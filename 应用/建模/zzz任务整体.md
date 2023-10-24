已有的几套代码：
- 3个文件 （配有 live2d, vrm ）
- mediapipe web 示例
- vtuber-python-unity, 理清楚了，回去跑一遍

TODO:
- [ ] 比对 3web demo sdk demo
- [x] 3web 能不能修改自由度
  -  添加自由度
- [ ] unity vtuber 复现
- [ ] 开始弄视频，先走 python + unity 路线
  - [ ] 修改计算特征的方式
  - [ ] 修改配置项
- [ ] 添加仿射变换等
- [ ] 封装上线

https://www.youtube.com/watch?v=NYuAU4QUjB8

## 目标

制作素材，将视频中的人物进行替换，或添加特效

## 流程

就是检测并驱动，但是细节非常多，尤其是驱动（运动学计算）这块，对不同的模型不同方式。

（驱动，也可改为配置项预设成固定的值）

检测并提取出特征，这块 mediapipe 都做了

后序的拼接也是挺麻烦的


## 问题

拆解问题，关键点检测，特征计算，（各种运动学计算、插值、很烦）配置项传送，做动画

如果mediapipe 处理好了，运动学只添加一个放射就行？ 插值，最终还要靠效果说话，正确处理的话应该能复刻素材的连贯性


- 生成动画的 抖动


## 方案比对

几种方案：
- web
- python + web
- python + unity
- python + live2d


`unity 开发应用`

python + unity + socket **当前主线**

live2d  对unity 支持得多，后续更换3d 模型，也方便

感觉 mediapipe js sdk 比 python sdk 简洁。。。

python + web 或 python + unity 收益都挺高的，目前还是 python + unity + socket 通信吧


按道理使用 web 可做到一个应用全部服务，但模型受限


`electron web`

当然，使用 web 对后续的素材管理等工作更容易衔接







## other

- 配置 expression, 不同的动画文件