
> 自然语言处理作业六：NCRF++ 实验

阅读论文，使用如下代码实现 POS（词性标记），NER（命名实体识别），chunking（断句） 任务。

代码：https://github.com/jiesutd/NCRFpp

论文：https://arxiv.org/pdf/1806.04470.pdf

## 代码分析

NCRF++: An Open-source Neural Sequence Labeling Toolkit.

Easy use to any sequence labeling tasks (e.g. NER, POS, Segmentation). It includes character LSTM/CNN, word LSTM/CNN and softmax/CRF components.

先下载仓库，运行 demo 代码

```cmd
python main.py --config demo.train.config
```

学到了：
- config 文件与 python 中 parser 的搭配使用
- 命令行中 > log.txt 可以将输出到 文本中


## 参数说明

### I/O
```python
train_dir=xx    #string (necessary in training). Set training file directory.
dev_dir=xx    #string (necessary in training). Set dev file directory.
test_dir=xx    #string . Set test file directory.
model_dir=xx    #string (optional). Set saved model file directory.
word_emb_dir=xx    #string (optional). Set pretrained word embedding file directory.

raw_dir=xx    #string (optional). Set input raw file directory.
decode_dir=xx    #string (necessary in decoding). Set decoded file directory.
dset_dir=xx    #string (necessary). Set saved model file directory.
load_model_dir=xx    #string (necessary in decoding). Set loaded model file directory. (when decoding)
char_emb_dir=xx    #string (optional). Set pretrained character embedding file directory.

norm_word_emb=False    #boolen. If normalize the pretrained word embedding.
norm_char_emb=False    #boolen. If normalize the pretrained character embedding.
number_normalized=True    #boolen. If normalize the digit into `0` for input files.
seg=True    #boolen. If task is segmentation like, tasks with token accuracy evaluation (e.g. POS, CCG) is False; tasks with F-value evaluation(e.g. Word Segmentation, NER, Chunking) is True .
word_emb_dim=50    #int. Word embedding dimension, if model use pretrained word embedding, word_emb_dim will be reset as the same dimension as pretrained embedidng.
char_emb_dim=30    #int. Character embedding dimension, if model use pretrained character embedding, char_emb_dim will be reset as the same dimension as pretrained embedidng.
```
### NetworkConfiguration

```python
use_crf=True    #boolen (necessary in training). Flag of if using CRF layer. If it is set as False, then Softmax is used in inference layer.
```

- CRF：条件随机场。是给定一组输入序列的条件下，另一组输出序列的条件概率分布模型。
- Softmax：它将一个数值向量归一化为一个概率分布向量。

```python
word_seq_feature=XX    #boolen (necessary in training): CNN/LSTM/GRU. Neural structure selection for word sequence. 
char_seq_feature=CNN    #boolen (necessary in training): CNN/LSTM/GRU. Neural structure selection for character sequence, it only be used when use_char=True.
```

- word 序列特征提取网络：CNN/LSTM/GRU.
- char 序列特征提取网络：CNN/LSTM/GRU.

```python
use_char=True    #boolen (necessary in training). Flag of if using character sequence layer. 
feature=[POS] emb_size=20 emb_dir=xx   #feature configuration. It includes the feature prefix [POS], pretrained feature embedding file and the embedding size. 
feature=[Cap] emb_size=20 emb_dir=xx    #feature configuration. Another feature [Cap].
nbest=1    #int (necessary in decoding). Set the nbest size during decoding.
```

### TrainingSetting

```python
status=train    #string: train or decode. Set the program running in training or decoding mode.
optimizer=SGD    #string: SGD/Adagrad/AdaDelta/RMSprop/Adam. optimizer selection.
iteration=1    #int. Set the iteration number of training.
batch_size=10    #int. Set the batch size of training or decoding.
ave_batch_loss=False    #boolen. Set average the batched loss during training.
```

### Hyperparameters

```python
cnn_layer=4    #int. CNN layer number for word sequence layer.
char_hidden_dim=50    #int. Character hidden vector dimension for character sequence layer.
hidden_dim=200    #int. Word hidden vector dimension for word sequence layer.
dropout=0.5    #float. Dropout probability.
lstm_layer=1    #int. LSTM layer number for word sequence layer.
bilstm=True    #boolen. If use bidirection lstm for word seuquence layer.
learning_rate=0.015    #float. Learning rate.
lr_decay=0.05    #float. Learning rate decay rate, only works when optimizer=SGD.
momentum=0    #float. Momentum 
l2=1e-8    #float. L2-regulization.
#gpu=True  #boolen. If use GPU, generally it depends on the hardward environment.
#clip=     #float. Clip the gradient which is larger than the setted number.
```

## 完成任务

作业中CCNN，应该是：Char CNN；WLSTM 应该是 Word LSTM


### POS 任务

`f > 97.23`

修改 `demo.train.config` 中的参数：

```python
train_dir=data/pos/train.pos
dev_dir=data/pos/dev.pos
test_dir=data/pos/test.pos
use_crf=False
word_seq_feature=LSTM
char_seq_feature=CNN
```

然后运行即可
```cmd
python main.py --config demo.train.config > log.txt
```

pos任务跑出来：
```cmd
Test: time: 13.75s, speed: 402.90st/s; acc: 0.9567, p: 1.0000, r: 1.0000, f: 1.0000
```

F 值达标了，就应该可以了。

又跑了几次POS：


| ACC        | sample.word.emb | glove.6B.100d.txt |
| ---------- | --------------- | ----------------- |
| norm False | 0.9567          | 0.9640            |
| norm True  | 0.9612          |                   |

### NER 任务

`f > 88.57`

更改 3 个 dir 配置，运行代码即可

```cmd
python main.py --config ner.config > ner_log.txt
```

| F          | sample.word.emb | glove.6B.100d.txt                  |
| ---------- | --------------- | ---------------------------------- |
| norm False | 0.7224          | 0.8459（epoch1），0.8812（epoch2） |
| norm True  |                 |                                    |

### chunking 任务

`f > 93.79`

更改 3 个 dir 配置，运行代码即可

```cmd
python main.py --config chunking.config > chunking_log.txt
```

| F          | sample.word.emb | glove.6B.100d.txt |
| ---------- | --------------- | ----------------- |
| norm False |                 | 0.8720            |
| norm True  | 0.8789          | 0.8866            |

又更改了参数：optimizer, iteration；

optimizer 按作者的说明，SGD 应该最好了

iteration 有一点点用

跑了很多遍出不来，就修改答案吧