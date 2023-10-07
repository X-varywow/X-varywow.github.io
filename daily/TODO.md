
https://www.fast.ai/

ML 学习路线 整合删除

- [ ] DCTNET 部署测试训练，就是最终方案；那个 sd_design 就挺好的效果，需要生成素材。（弄好，再抠图拼接）
- [ ] 补算法， p2p 等；浏览器，UNET；图像翻译 blog

UNET

[深度学习之图像翻译与风格化GAN-理论与实践](https://www.bilibili.com/video/BV1Wr4y1b77B)

- [ ] DFL 开始跑
- [ ] 最多训 2 版大的 DFL
- [ ]  python statsmodels

bili 论文解读

- [ ] 整理归档，ASR 课程笔记
- [ ] VAE 完善
- [ ] 中文 tts 搭建，4080
- [ ] https://www.zhihu.com/people/new-iron/zvideos

https://tianchi.aliyun.com/competition/entrance/531961/rankingList


https://www.tonyisstark.com/1708.html


beautyGAN 实现 脸型调整，妆造迁移


- opencv https://wizardforcel.gitbooks.io/py-ds-intro-tut/content/opencv.html
- 只狼强化学习 https://www.bilibili.com/video/BV1by4y1n7pe/
- qlearning  https://github.com/enhuiz/flappybird-ql
- 手写一个 VGG
- 弄个京东自动抢购的，茅
- chatGPT，弄个机器人；new bing
- 弄个关键词，自动写诗的？
- genshin script
- .NET Framework
- 菜鸟教程
- electron开发，代替 TODO，自定义背景，等
- 合理键盘




- [ ] 强化学习 genshin win32 如何置顶，助手，不同 action 打分，策略+模型
- [ ] 右侧添加一个单页面的 目录
- [ ] 整理 torch 文档到 blog
- [ ] python 数据处理

- freecodecamp
  - [ ]  JavaScript Algorithms and Data Structures
  - [ ]  Legacy Responsive Web Design
  - [ ]  Machine Learning with Python

- [ ] opensearch 日志平台
- [ ] 6字节后端
- [ ] 学习 alpha 公开的 MuZero 算法
- [ ] https://github.com/zhiyiYo/Alpha-Gobang-Zero
- [ ] 有空多去看 pytorch 文档
- [ ] bark

免费的GPT:
https://github.com/xiangsx/gpt4free-ts

https://github.com/linyiLYi/snake-ai 用于 ML 动手的小实践， CNN


LIVE2D 做动画，软件硬件，GOGOGO；加入搞机玩家


--------------------


https://khoj.dev/

pixie+deca


前端应用程序开发：https://github.com/Moonvy/OpenPromptStudio


https://docs.warudo.app/warudo/v/en/mocap/mediapipe 集大成者

https://github.com/nladuo/live2d-chatbot-demo



[MLP-Mixer](https://keras.io/examples/vision/mlp_image_classification/#the-mlpmixer-model)


近大远小，类似这个：
https://developers.googleblog.com/2020/08/instant-motion-tracking-with-mediapipe.html


https://github.com/ardha27/AI-Waifu-Vtuber/



wallpaper 原理，unity 无边框全屏


- [ ] blog - 0822vtuber

https://github.com/wangshub

https://jalammar.github.io/illustrated-transformer/


https://zhuanlan.zhihu.com/p/54356280

https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3

http://nlp.seas.harvard.edu/2018/04/03/attention.html


https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

https://blog.csdn.net/Raina_qing/article/details/106374584

https://zhuanlan.zhihu.com/p/558937247

https://yang-song.net/blog/2021/score/

深度学习入门


- [ ] 找特效贴图类代码做参考，3web 是另一套
- [ ] https://www.cloudskillsboost.google/journeys/118


- [ ] 尝试桌面AI, unity(/electron) + python + vits + llama + live2d 动画（事件响应）(flask 作为后端)
- [ ] genshin 脚本，素材训练
- [ ] 比对 vits silero

https://github.com/RimoChan/Vtuber_Tutorial

[虚拟桌宠模拟器](https://github.com/LorisYounger/VPet) csharp

- [ ] 脚本
- [ ] vt
- [ ] 语音
- [ ] https://windland-neotix.vercel.app/ +  chatgpt + 重开




https://developers.googleblog.com/2023/05/introducing-mediapipe-solutions-for-on-device-machine-learning.html

考虑自己写整个流程，别人的效果不好，旧的。

https://codepen.io/mediapipe/details/KKgVaPJ

https://github.com/google/mediapipe/blob/v0.9.1/docs/solutions/face_mesh.md Face Effect Example


- [ ] 高级算法，卡尔曼滤波，bili 学一下
- [ ] 下载：https://github.com/yeemachine/kalidoface/releases，（这个效果应该是最好了，）用笔记本（或手机当摄像头）


- [ ] 3web 做成应用，测试一下。
- [ ] 更新 3web


https://blog.csdn.net/qq_42139931/article/details/122038862 VRM 模型导出

https://github.com/LorisYounger/VPet


https://www.live2d.com/zh-CHS/download/sample-data/ 中有个通过形状动画扩展模型的可动区域，实现不依赖参数的自由动作。


## 语音

[silero-models](https://github.com/snakers4/silero-models) 文本转语音，语音转文本；还支持 ssml；

声音非常真实，音质很高，语气较丧。在 cpu 也能很快推理。可见，使用好的数据集会有很好的效果。

轻量级，gpt 说 vits 效果更好

[colab demo](https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples_tts.ipynb)


https://github.com/Plachtaa/VALL-E-X

语音最近也要迭代一下，（valle vits 几个对比，整理）



改进 audio-service


```python

# svc 中 load 实现
def load_model(self):
    # get model configuration
    self.net_g_ms = SynthesizerTrn(
        self.hps_ms.data.filter_length // 2 + 1,
        self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
        **self.hps_ms.model)
    _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
    if "half" in self.net_g_path and torch.cuda.is_available():
        _ = self.net_g_ms.half().eval().to(self.dev)
    else:
        _ = self.net_g_ms.eval().to(self.dev)


# unload 在 Svc 中实现
model.unload_model()
model = None
torch.cuda.empty_cache()
return sid.update(choices = [],value=""),"模型卸载完毕!"


# vits 实现方式？
del net_g
gc.collect()
```

transformer
- https://mp.weixin.qq.com/s/gvL6CjQWzhI5hBclBZk2qA