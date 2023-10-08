VALL-E 把训练数据拔高到了 60000 小时，模型方面使用了 96 层 TransformerDecoder 结构，Zero-short TTS 能力相比之前工作提升明显;

这里的 **zero-shot** 指的是无需 fine-tuning 模型参数，通过提供 prompt text 和 audio 来实现声音克隆。


[MS官方效果展示](https://www.microsoft.com/en-us/research/project/vall-e-x/)，没有开源

----------

|                                                               | star | 说明                                                                                   |
| ------------------------------------------------------------- | ---- | -------------------------------------------------------------------------------------- |
| [lifeiteng复现](https://github.com/lifeiteng/vall-e)          | 1.4k | [复现效果展示](https://lifeiteng.github.io/valle/index.html)                           |
| [enhuiz复现](https://github.com/enhuiz/vall-e)                | 2.7k |                                                                                        |
| [Plachtaa复现 VALL-E X](https://github.com/Plachtaa/VALL-E-X) | 5.7k | 20231008 比较新的模型 zero-shot，但业务上 zero-shot 用不太上，单论质量达不到生产需求。 |

