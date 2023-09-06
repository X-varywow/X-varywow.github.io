

(1) [VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)

[colab demo](https://colab.research.google.com/drive/1pn1xnFfdLK63gVXDwV4zCXfVeo8c-I-0)

参考左侧目录

</br>

(2) [microsoft/speecht5_tts](https://github.com/microsoft/SpeechT5)

参考左侧目录

</br>


(3) [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) 39.6k

参考左侧目录 RTVC

中文版本：[MockingBird](https://github.com/babysor/MockingBird) 

</br>

(4) 工具类

https://github.com/coqui-ai/TTS  9.1K [官网](https://coqui.ai/)

[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) 6.6K

</br>

(5)
[语音合成（TTS） 论文优选：Emotion TTS](https://zhuanlan.zhihu.com/p/302276645)

https://silyfox.github.io/iscslp-98-demo/

</br>


(6) VALL-E X

[Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling](https://arxiv.org/abs/2303.03926)

[demo](https://vallex-demo.github.io/#emotion-samples), 可以情感控制

</br>

(7)
[语音合成论文优选：Towards Multi-Scale Style Control for Expressive Speech Synthesis](https://blog.csdn.net/liyongqiang2420/article/details/115672959)

https://arxiv.org/pdf/2104.03521.pdf

</br>

(8) fastspeech2

参考 [huggingface demo](https://huggingface.co/facebook/fastspeech2-en-ljspeech)

效果不如 VITS

</br>

(9) https://github.com/PlayVoice/vits_chinese  

[demo](https://huggingface.co/spaces/maxmax20160403/vits_chinese/tree/main)

Best TTS based on BERT and VITS with some Natural Speech Features Of Microsoft; Also for voice clone!

1, Hidden prosody embedding from BERT，get natural pauses in grammar

2, Infer loss from NaturalSpeech，get less sound error

3, Framework of VITS，get high audio quality

</br>

(10) [VITS-派蒙](https://www.bilibili.com/video/BV16G4y1B7Ey/)

</br>

(11) [Sovits](https://github.com/Francis-Komizu/Sovits) 

An implementation of the combination of Soft-VC and VITS

[sovits4.0一键训练/推理脚本.ipynb](https://colab.research.google.com/drive/1hGt9XowC07NGmXxKNJvY5N64uMdd435M)

[soft_vc_demo](https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb)

https://github.com/svc-develop-team/so-vits-svc ⭐️

https://www.bilibili.com/video/BV1H24y187Ko/ (纯工程应用向，实用性强)

1-2h 高质量声音即可

so-vits 社区挺牛的，0417 最近还在更新，实用性强

</br>

（12）重风格模型

sampleRNN? 

没有语义内容，但保留了发音、音色以及韵律风格

WaveNet？

https://unisound.github.io/end-to-end_tts/


(13) [emotional-vits](https://github.com/innnky/emotional-vits) 

无需情感标注的情感可控语音合成模型，基于VITS

[colab vits模型训练笔记本-添加情感向量支持](https://colab.research.google.com/drive/10MkPCQhhTs30jwUSMpZ8mTbptqpUOLnl?usp=sharing#scrollTo=LOsV22D8IUTS)

[inference.ipynb](https://github.com/innnky/emotional-vits/blob/main/inference.ipynb)

[MoeGoe.exe](https://github.com/CjangCjengh/MoeGoe)


（14）https://github.com/dtx525942103/vits_chinese

（15）[bark](https://github.com/suno-ai/bark)

Spear-TTS 的路线，AudioToken 预测上借鉴了 VALL-E.

[colab demo](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)

infer 比其他模型慢很多（两句话 5min），应该不是 colab 的问题；

效果不好（自己部署的，huggingface上的），文档不全

（16）[AudioGPT](https://github.com/AIGC-Audio/AudioGPT)

（17）[voice-changer](https://github.com/w-okada/voice-changer)


## Other


> 不感兴趣的话，模型的调研和改进是十分耗时的事情，不如直接用这些：

- https://revoicer.com/
- https://ttsmaker.com/
- https://voicemaker.in/
- [elevenlabs](https://beta.elevenlabs.io/)
- [标贝科技](https://www.data-baker.com/)
- https://play.ht/


----------

other：

[dl-for-emo-tts](https://github.com/Emotional-Text-to-Speech/dl-for-emo-tts)， A summary on our attempts at using Deep Learning approaches for Emotional Text to Speech

[Expressive-FastSpeech2](https://github.com/keonlee9420/Expressive-FastSpeech2)
duration, pitch, energy

[Controllable Emotion Transfer For End-to-End Speech Synthesis](https://blog.csdn.net/qq_40168949/article/details/123615686)

[Expressive TTS 相关论文阶段性总结](https://blog.csdn.net/qq_35668477/article/details/115588528)


小结：确实没有什么情感可控的模型，情感作为微调参数 or 重新训练?
还是要先学一下那个 vits-fine-tune，不管是 VITS 基础架构，torch 训练神经网络的方式，还是 video2wav 等各种处理，模型如何微调，，，（0316 还是先尝试）


小结：感觉大公司做的 TTS 都不太行啊，还没小公司或开源的 vits 好。工作中心转移到 VITS 和数据集上了。