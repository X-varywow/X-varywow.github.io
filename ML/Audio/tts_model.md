
_理论模型_

- `WaveNet` (2016)
- `Tacotron` (2017)
  - towards end_to_end speech synthesis
  - [简要介绍](https://weikaiwei.com/neural/tacotron/)


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230402183927.png">


- `Tacotron2` (2017)
  - Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions
  - 结构
    - 文本 -> mel spectrogram; 采用了经典的 encoder - attention - decoder
    - mel spectrogram -> speech wav; 使用 vocoder
- `GAN-TTS` (2019)
- `FastSpeech 2` (2020)
  - Fast and High-Quality End-to-End Text to Speech
- `VITS` (2021)
  - [论文地址](https://arxiv.org/abs/2106.06103) [Paperwithcode](https://cs.paperswithcode.com/paper/conditional-variational-autoencoder-with)
  - [细读经典：VITS，用于语音合成带有对抗学习的条件变分自编码器](https://zhuanlan.zhihu.com/p/419883319)
  - [VITS;  完全端到端TTS的里程碑](https://blog.csdn.net/Terry_ZzZzZz/article/details/120458064)
  - [论文复现1](https://github.com/jaywalnut310/vits) 2.6k


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230402183929.png">


- `SyntaSpeech` (2022)
  - Syntax-aware Generative Adversarial Text-to-Speech; IJCAI 2022;
  - [展示界面](https://syntaspeech.github.io/)
  - [论文地址](https://arxiv.org/abs/2204.11792)
  - [hugging face](https://huggingface.co/spaces/yerfor/SyntaSpeech)
- `GenerSpeech` (2022)
  - [github](https://github.com/Rongjiehuang/GenerSpeech)
  - [论文地址](https://arxiv.org/abs/2205.07211)
- `YourTTS` (2022)
  - [audio samples](https://edresson.github.io/YourTTS/)
  - [github repo](https://github.com/Edresson/YourTTS) 
  - zero-shot 指的是无需 fine-tuning 模型参数，通过提供 prompt text 和 audio 来实现声音克隆。
- `MsEmoTTS` (2022)
  - Multi-scale emotion transfer, prediction, and control for emotional speech synthesis
  - 多个层级，全局、句子、局部，对情感表征
  - [论文地址](https://arxiv.org/abs/2201.06460)
  - [智能语音技术新发展与发展趋势](https://blog.csdn.net/soaring_casia/article/details/122303288)
  - [Expressive TTS：向更有表现力的语音合成进发（一）](https://zhuanlan.zhihu.com/p/133388563)
- `NaturalSpeech 2` (2023)
  - [论文地址](https://arxiv.org/abs/2304.09116)
- `VALL-E` (2023) microsoft
  - [展示界面](https://valle-demo.github.io/)
  - [论文地址](https://arxiv.org/abs/2301.02111)
  - [实现1](https://github.com/lifeiteng/vall-e) [实现2](https://github.com/enhuiz/vall-e)
- `SPEAR-TTS` (2023) google
  - [展示界面](https://google-research.github.io/seanet/speartts/examples/)
  - [实现1(未完全)](https://github.com/collabora/spear-tts-pytorch)


</br>


_细分模型_

- `encoder`
  - [huBERT](https://github.com/bshall/hubert)
- `vocoder` (mel-spectrograms -> speech)
  - hifiGAN
  - WaveGlow (a flow-based network)
- `synthesizer`

</br>

_预训练模型_

- https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2
  - ljs 139mb
  - vctk 150mb

</br>

-------------

参考资料：

- https://paperswithcode.com/methods/category/text-to-speech-models
- https://paperswithcode.com/task/emotional-speech-synthesis
