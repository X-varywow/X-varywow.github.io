
## preface

[课程地址](https://www.youtube.com/playlist?list=PLJV_el3uVTsO07RpBYFsXg-bN5Lu0nhdG)

综述性课程，需要一定基础，不太会

## Speech recognition

1 second has 16k sample points, each point has256 possible values

Seq2Seq（Sequence to sequence）模型，是将序列（Sequence）映射到序列的神经网络机器学习模型。 这个模型最初设计用于改进机器翻译技术，可容许机器通过此模型发现及学习将一种语言的语句（词语序列）映射到另一种语言的对应语句上。

以前的语音识别比较复杂，用的 seq2seq，但现在用深度学习 “硬Train一发” 即可。

Token（按 ASR 使用排行）
- Grapheme, smallest unit of a writing system
- Phoneme, a unit of sound
- Morpheme, the smallest meaningful unit
- Word

Acoustic Feature（按 ASR 使用排行）
- filter bank output
- mfcc
- waveform
- spectrogram

Waveform ---DFT--> spectrogram， [参考wiki](https://zh.wikipedia.org/zh-hans/%E6%97%B6%E9%A2%91%E8%B0%B1)

----> filter bank ----> log ----> DCT ----> MFCC(梅尔倒频谱)

---------

- LAS, Listen, Attend, and Spell
- CTC, Connectionist Temporal Classification
- RNN Transducer
- Neural Transducer
- Monotonic Chunkwise Attention

## Voice Conversion

cycle GAN



## Speech Synthesis

文法信息、情感系统、停顿，More information for encoder;

[论文阅读笔记：Tacotron和Tacotron2](https://zhuanlan.zhihu.com/p/337042442)


Controllable TTS

voice cloning，语调、重音、韵律、抑扬顿挫


[vocoder](https://www.youtube.com/watch?v=6g2aPc0ol2Y), 将 spectrogram 变成声音信号。


other:
- Speaker Verification


## DL

https://speech.ee.ntu.edu.tw/~hylee/honor.php

https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php


