
## _数据集_


| 数据集   | 大小                | 说明             | 地址                                                    |
| -------- | ------------------- | ---------------- | ------------------------------------------------------- |
| MNIST    |                     | 手写数字训练集   |                                                         |
| ImageNet | 120万张             |                  |                                                         |
| COCO     |                     |                  | [官网](https://cocodataset.org/)                        |
| FFHQ     |                     | 高清人脸数据集   | [github](https://github.com/NVlabs/ffhq-dataset)        |
| VoxCeleb |                     | 人像、语音数据集 | [官网](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| VGGFace2 | 3.3 million + faces |                  |                                                         |
|          |                     |                  |                                                         |





参考资料：
- https://zhuanlan.zhihu.com/p/129736067




## _卡通图像翻译_

简单的卡通风格，门神风格素材：https://github.com/justinpinkney/toonify

317张 cartoon: https://mega.nz/file/HslSXS4a#7UBanJTjJqUl_2Z-JmAsreQYiJUKC-8UlZDR0rUsarw

动漫素材1：https://gwern.net/crop#portraits-dataset

```bash
# 16gb 的素材
rsync--verbose--recursive rsync://176.9.41.242:873/biggan/portraits/ ./portraits/
```

https://github.com/learner-lu/anime-face-dataset


较好的数据集：https://www.kaggle.com/datasets/lukexng/animefaces-512x512


[切图工具](https://github.com/nagadomi/lbpcascade_animeface)


-------------


构造的数据集1：
偏迪斯尼的卡通风格

