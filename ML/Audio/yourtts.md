
## preface

YourTTS为TTS带来了zero-shot多说话人和多语言模式。方法建立在 VITS 上，并争对多说话人和多语言做了些调整。

VTCK上取得了SOTA效果，在zero-shot multi-speaker TTS中的结果与SOTA相当。

zero-shot 指的是无需 fine-tuning 模型参数，通过提供 prompt text 和 audio 来实现声音克隆。

可以用不到 1 分钟的语音微调 YourTTS 模型，并在语音相似性和合理质量上取得了最先进的结果。


## model

[hugging online demo](https://huggingface.co/spaces/ICML2022/YourTTS)

5s, 20s, 150s 都试了一下, 不太行，噪声比 vits 大，学到的音色、停顿并不太像。

是利用 speaker embedding 的都会有噪声这个问题吗，那个 speech t5 也是， 但是 t5 的音色、停顿比这个好多了。

colab 上也跑了一遍，**使用 speaker embedding 的不靠谱**。还是要 train


---------

参考资料：
- [audio samples](https://edresson.github.io/YourTTS/)
- [github repo](https://github.com/Edresson/YourTTS) 
- [知乎 - yourtts 介绍](https://zhuanlan.zhihu.com/p/599275055)