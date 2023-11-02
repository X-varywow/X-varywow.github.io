
_数据集_

| 数据集                                                       | 语种 | 说话人 | 大小  | 时长/人 | 采样率（kHz） | 备注                                                  |
| ------------------------------------------------------------ | ---- | ------ | ----- | ------- | ------------- | ----------------------------------------------------- |
| [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)         | en   | 1      | 2.6GB | 24h     | 22.05         |                                                       |
| [LibriTTS](https://www.openslr.org/60/)                      | en   | 2456   |       | 4.2h    | 24            | 只有16kHz                                             |
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)         | en   | 109+1  | 11GB  | 0.4h    | 48            | 总时长41.6h，说话人都是年轻人，感觉这个数据集不太高兴 |
| [aidatatang_200zh](https://openslr.org/62/)                  | ch   |        | 18GB  |         |               |                                                       |
| [Hi-Fi Multi-Speaker](https://www.openslr.org/109/)          | en   | 11     | 41GB  | 27.1h   | 44.1          |                                                       |
| [RyanSpeech](https://huggingface.co/datasets/Roh/ryanspeech) | en   | 1      |       | 10h     | 44.1          |                                                       |


</br>

linux 获取代码：

```bash
# ljsspeech 数据集
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -jxvf LJSpeech-1.1.tar.bz2

# vctk 数据集
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip
sudo yum install p7zip
7za x DS_10283_3443.zip
7za x VCTK-Corpus-0.92.zip
# 对于 zip 文件，还可以使用 unzip，不适用于大文件？


# hifi 数据集
wget https://www.openslr.org/resources/109/hi_fi_tts_v0.tar.gz
tar -zxvf hi_fi_tts_v0.tar.gz


# how to get ryan speech
# https://huggingface.co/datasets/Roh/ryanspeech/blob/main/ryanspeech.py
wget https://huggingface.co/datasets/Roh/ryanspeech/resolve/main/data/train.tar.gz
tar -zxvf train.tar.gz

# 对于 7z 文件
sudo yum install p7zip -y
7za x contentvec768l12.7z
```
------------

参考资料：
- http://yqli.tech/page/data.html
- [更多数据集](https://zhuanlan.zhihu.com/p/267372288)
- [AISHELL-2：全球最大中文开源数据库](https://cloud.tencent.com/developer/news/249984)
- [What makes a good TTS dataset](https://github.com/coqui-ai/TTS/wiki/What-makes-a-good-TTS-dataset)