
TTS, Text-To-Speech。 

常用技术路线：（1）拼接法 （2）参数法


</br>

参数TTS系统可分为两大模块：前端和后端。
- `前端` 包含文本正则、分词、多音字预测、文本转音素和韵律预测等模块，
  - 它的功能是把输入文本进行解析，获得音素、音调、停顿和位置等语言学特征。 
- `后端` 包含时长模型、声学模型和声码器，
  - 它的功能是将语言学特征转换为语音。
  - 时长模型的功能是给定语言学特征，获得每一个建模单元（例如:音素）的时长信息；
  - 声学模型则基于语言学特征和时长信息预测声学特征；
  - 声码器则将声学特征转换为对应的语音波形。

> 相较于后端，前端更加庞杂，包含分词、韵律分析、注音模块等，但是学术界的主要精力集中在后端，因为更容易发paper。前端的研究相对较少，而中文前端更甚。

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230830222946.jpg" style="zoom:50%">


试了一圈下来，bark, vits, t5, tactron, fastspeech, valle 最后用的 vits + sovits


-------------------

参考资料：
- [Sambert-Hifigan模型介绍](https://modelscope.cn/models/damo/speech_sambert-hifigan_nsf_tts_cally_en-us_24k/summary)