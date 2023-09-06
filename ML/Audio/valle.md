VALL-E 把训练数据拔高到了 60000 小时，模型方面使用了 96 层 TransformerDecoder 结构，Zero-short TTS 能力相比之前工作提升明显;

这里的 **zero-shot** 指的是无需 fine-tuning 模型参数，通过提供 prompt text 和 audio 来实现声音克隆。


[官方效果展示](https://valle-demo.github.io/)


[复现1](https://github.com/lifeiteng/vall-e) [复现效果展示](https://lifeiteng.github.io/valle/index.html)

[复现2](https://github.com/enhuiz/vall-e)
