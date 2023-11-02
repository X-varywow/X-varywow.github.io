

## Preface


1，VITS 框架包含两个子系统：基于 VAE 的变声系统以及基于 Flow 的语音合成系统；VAE 擅长捕捉句子整体的韵律特征，而 Flow 擅长重建音频的细节特征；将两者整合，进行多任务训练，实现参数与优势共享。

2，VITS 的语音合成系统直接合成音频而不是 MEL 谱，实现真正意义的端到端语音合成，而非分为两个模型（声学模型&声码器）的序列到序列的合成；从而消除两个模型带来的 Gap。

3，传统两个模型的 TTS 系统，GAN 训练通常只应用与声码器；而，VITS 中的 GAN 训练是全局的、对每个模块都有效。

----------


一些名词：stochastic 随机分布的; posterior 后部的




## 文件结构


- `configs` (json 配置信息)
  - vctk_base.json
    - train，log_interval, eval_interval, epochs等训练 config
    - data，描述音频数据相关信息，train 路径、vali 路径、sample rate 等
    - model，描述模型参数，inter_channels, hidden_channles, n_heads, n_layers 等
  - ljs_base.json
- `filelists` (模型需要的数据，指明了 wav 路径，对应的 phoneme)
  - test
  - train
    - txt
    - cleaned (这是 txt 对应的音素)
  - val
- monotonic_align
  - 使用 cpython 实现的单调对齐搜索
- text
  - 负责 filelists 的处理
  - 特殊文本符号的处理
- attentions.py
  - encoder, decoder, MultiHeadAttention, FFN
- commons.py
  - init_weights, get_padding
- `data_utils.py`
  - textaudioloader
  - TextAudioCollate
  - TextAudioSpeakerLoader
  - DistributedBucketSampler，构造分桶数据
- losses.py
  - feature_loss, discriminator_loss, generator_loss, kl_loss
- mel_processing.py
  - 频谱图信息处理
- `models.py`
  - StochasticDurationPredictor 随机时长预测器
  - DurationPredictor
  - TextEncoder 先验编码器（包含文本编码器和标准化流）
  - ResidualCouplingBlock 标准化流
  - PosteriorEncoder
  - Generator
  - DiscriminatorP
  - DiscriminatorS
  - MultiPeriodDiscriminator
  - SynthesizerTrn
- `modules.py`
  - LayerNorm, ConvReluNorm, DDSConv, WN, ResBlock, ConvFlow 等
- `preprocess.py`
  - 处理训练数据文本
- `train.py, train_ms.py`
- transforms.py
  - piecewise_rational_quadratic_transform
  - unconstrained_rational_quadratic_spline
  - rational_quadratic_spline
- utils.py
  - load_checkpoint, save_checkpoint, plot 等


## 训练流程

[colab vits demo](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf)

```bash
git clone https://github.com/jaywalnut310/vits.git

cd vits
pip install -r requirements.txt

sudo yum install espeak -y

cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

训练先要对 音频处理 ，参考 README ， preprocess.py

```bash
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 

python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```

总体：音频原始数据固定格式保存，然后音频处理保存音素filelist。配置 config, 准备好 mp3 和 txt 就可以开始训练了 



额外：


```bash
pip install gdown

# pretrained_vctk.pth (152M)
gdown 'https://drive.google.com/uc?id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru'

# pretrained_ljs.pth (139M)
gdown 'https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT'
```

```bash
# ljsspeech 数据集
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -jxvf LJSpeech-1.1.tar.bz2

# vctk 数据集
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip
sudo yum install p7zip
7za x DS_10283_3443.zip
7za x VCTK-Corpus-0.92.zip
```




-----------

使用 ryan 数据集，处理并生成训练数据（用到 whisper 做语音识别）：


</br>

_STEP1_
```python
from pathlib import Path
from tqdm import tqdm
import torchaudio
import os
import whisper
from torch.utils.data import random_split
import torch

DEV_PATH = Path("/home/ec2-user/SageMaker/dev")
TRAIN_PATH = Path("/home/ec2-user/SageMaker/train")
TEST_PATH = Path("/home/ec2-user/SageMaker/test")

def show_infos():
    dev = list(DEV_PATH.glob("**/*.wav"))
    train = list(TRAIN_PATH.glob("**/*.wav"))
    test = list(TEST_PATH.glob("**/*.wav"))
    print(f"dev:{len(dev)};train:{len(train)};test:{len(test)}")
    return train, test, dev
    
show_infos()
```

</br>

_STEP2_
```python
# use py and cmd please

def audio2text(after):
    model = whisper.load_model("base")
    train, test, dev = show_infos()
          
    with open(after+"ryan_train.txt", "w") as f:
        for file in tqdm(train, desc="train"):
            file = str(file)
            text = model.transcribe(file)["text"].strip()
            f.write(file + "|" + text + "\n")
        
        
    with open(after+"ryan_test.txt", "w") as f:
        for file in tqdm(test, desc="test"):
            file = str(file)
            text = model.transcribe(file)["text"].strip()
            f.write(file + "|" + text + "\n")
      
        
    with open(after+"ryan_val.txt", "w") as f:
        for file in tqdm(dev, desc="dev"):
            file = str(file)
            text = model.transcribe(file)["text"].strip()
            f.write(file + "|" + text + "\n")

            
            
if __name__ == "__main__":       
    audio2text("/home/ec2-user/SageMaker/vits/filelists/")
```

</br>

_STEP3_
```bash
# use py and cmd

python preprocess.py --text_index 1 --filelists filelists/ryan_train.txt filelists/ryan_test.txt filelists/ryan_val.txt
```

```bash
# change json and train.py
# batchsize 20, and filelist use ryan
# gpu V100 16GB

python train.py -c configs/ryan_base.json -m ryan_base
```

## webui

```python
import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import gradio as gr


# 固定来下
# hps.data.text_cleaners 总是 ["english_cleaners2"]
# hps.data.add_blank 总是 true
def get_text(text):
    text_norm = text_to_sequence(text, ["english_cleaners2"])
    # print(text_norm)
    
    # if hps.data.add_blank:
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
```

```python

  
## MODEL1 ljs_speech 
hps_ljs = utils.get_hparams_from_file("./configs/ljs_base.json")
model_ljs = SynthesizerTrn(
    len(symbols),
    hps_ljs.data.filter_length // 2 + 1,
    hps_ljs.train.segment_size // hps_ljs.data.hop_length,
    **hps_ljs.model)
# eval() set the module evaluation mode. 影响 dropout, batchnorm 层
model_ljs.eval()
# let model load_state_dict from pth
utils.load_checkpoint("pretrained_ljs.pth", model_ljs, None)


## MODEL2 vctk multi speaker
hps_ms = utils.get_hparams_from_file("./configs/vctk_base.json")
model_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
model_ms.eval()
utils.load_checkpoint("pretrained_vctk.pth", model_ms, None)


## MODEL3 ryan
hps_ryan = utils.get_hparams_from_file("./configs/ryan_base.json")
model_ryan = SynthesizerTrn(
    len(symbols),
    hps_ryan.data.filter_length // 2 + 1,
    hps_ryan.train.segment_size // hps_ryan.data.hop_length,
    **hps_ryan.model)
model_ryan.eval()
utils.load_checkpoint("./logs/ryan_base/G_253000.pth", model_ryan, None)


d = {
    "base(female)": 1001,
    "base(male)": 1002,
    "male1": 25,
    "male2": 16,
    "male3": 9,
    "female1": 22,
    "female2": 84,
    "female3": 48
}


def tts_fn(text: str, speaker: str, speed, noise_scale, noise_scale_w):
    speaker = d[speaker]
    stn_tst = get_text(text)
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    
    if speaker == 1001:
        # Disabling gradient calculation is useful for inference,
        with torch.no_grad():
            audio = model_ljs.infer(x_tst, x_tst_lengths, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1.0/speed)[0][0,0].data.float().numpy()
    elif speaker == 1002:
        with torch.no_grad():
            audio = model_ryan.infer(x_tst, x_tst_lengths, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1.0/speed)[0][0,0].data.float().numpy()
    else:
        sid = torch.LongTensor([speaker]) # speaker identity
        with torch.no_grad():
            audio = model_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1.0/speed)[0][0,0].data.float().numpy()

    return "Success", (hps.data.sampling_rate, audio)


app = gr.Blocks()
with app:
    with gr.Row():
        with gr.Column():
            text  = gr.Textbox(label = "Text",
                               placeholder = "Type your sentence here",
                               value = SHOW_TEXT,
                               elem_id = "tts-input")

            speaker = speaker = gr.Radio(label="Speaker", choices=["base(female)", "base(male)", "male1","male2","male3",
                                                                  "female1","female2"], value="base(female)")
            
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="语速")
            noise_scale = gr.Slider(minimum=0.1, maximum=1, value=0.667, step=0.001, label="情感变化")
            noise_scale_w = gr.Slider(minimum=0.1, maximum=1, value=0.8, step=0.1, label="音素发音长度")
            
        with gr.Column():
            text_output = gr.Textbox(label="Message")
            audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
            btn = gr.Button("Generate!")
            btn.click(tts_fn,
                      inputs=[text, speaker, speed, noise_scale, noise_scale_w],
                      outputs=[text_output, audio_output])
            
app.launch()
    
```

## 代码细分(tmp)

从训练来看

```python
# train.py

def main():

hps 获取配置信息

torch.multiprocessing

  def run():
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.set_device(rank)
    TextAudioLoader
    DistributedBucketSampler 将数据分桶，近似长度的在一个桶
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)

    # 创建网络
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    # AdamW 优化器
    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)

    # DDP 分布训练
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    #指数衰减调整学习策略
    torch.optim.lr_scheduler.ExponentialLR

    # 混合精度，提高训练速度
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch:
      train_and _evalute()
      scheduler.step()

def train_and_evaluate():
  net.train()

  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
    with autocast():
      mel
      y_mel
      y_hat_mel
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      discriminator_loss(y_d_hat_r, y_d_hat_g)





```


MultiPeriodDiscriminator 判别网络

SynthesizerTrn 合成网络













## bert_vits

后端声学模型使用的 bert

https://github.com/fishaudio/Bert-VITS2









## 其他说明

文本 -> 频谱 -> 波形

文本 -> 波形

end_to_end，使用 <text, audio> pairs 来训练模型;


noise_scale 控制感情变化

noise_scale_w 控制音素发音长度 （VITS 学到了说话人相关的音素时长）

length_scale 控制整体语速, 越大长度越长，语速越慢


VITS 是一个由音素直接映射为波形的端到端模型，

使用 [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/tree/4.0-v2) 进行声音克隆，[hugging face demo](https://huggingface.co/spaces/zomehwh/sovits-models)


多线程 run(process_idx, *args)
main() -> torch.multiprocessing.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

```python
# 断点训练
def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x
```

一次 log-info:

```
119000.pth

loss_disc: 2.3796546459198    # discriminator_loss

loss_gen, 2.3140931129455566  # generator_loss

loss_fm, 5.803615093231201     # feature_loss

loss_mel, 22.382572174072266  # F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel

loss_dur, 1.5575270652770996   # torch.sum(l_length.float())

loss_kl,  1.8198809623718262   # kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
 
global step: 119200,

lr: 0.00019754009828923033
```



------------

参考资料
- [原作者码仓](https://github.com/jaywalnut310/vits)
- [细读经典：VITS，用于语音合成带有对抗学习的条件变分自编码器](https://zhuanlan.zhihu.com/p/419883319)
- [知乎；vits 发展历程](https://zhuanlan.zhihu.com/p/474601997)
- [COQUI AI 的实现](https://github.com/coqui-ai/TTS/blob/f237e4ccd9f2fd2b0cb5e136dfe0e20cc32bf898/TTS/tts/models/vits.py)
- [VITS论文阅读](https://blog.csdn.net/zzfive/article/details/127061469)
  - [vits官方gituhb项目--数据处理](https://blog.csdn.net/zzfive/article/details/127336473)
  - [vits官方gituhb项目--模型训练](https://blog.csdn.net/zzfive/article/details/127503913)
  - [vits官方gituhb项目--模型构建](https://blog.csdn.net/zzfive/article/details/127540768)
- [基于cVAE+Flow+GAN的效果最好语音合成VITS模型代码逐行讲解](https://www.bilibili.com/video/BV1VG411h75N/)