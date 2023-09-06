
> 通过这个项目，可以实现中等水平的 语音克隆、语音生成；

## Preface

[huggingface: speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)

[在线demo](https://huggingface.co/spaces/Matthijs/speecht5-tts-demo)

如何使用：

```python
!pip install git+https://github.com/huggingface/transformers sentencepiece datasets
!pip install gradio
!pip install soundfile
!pip install speechbrain

## vc
!pip install inflect==5.3.0
!pip install librosa
!pip install visdom==0.1.8.9
!pip install webrtcvad==2.0.10
```


```python
# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)

```

>随着 samplerate 变高，语速变快，音高变高


## 音色相关

[huggingface: cmu-arctic-xvectors](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors)

如何使用：

```python
from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

speaker_embeddings = embeddings_dataset[7306]["xvector"]
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

speaker_embeddings.shape
# -> torch.Size([1, 512])

# embeddings_dataset 是个字典组成的列表，[{'filename':,'xvector':[]},{},,,]
```

音色说明:
- 7306，压抑女声
- 5793，1605， 3592 正常男
- 3369，7355 正常女
- 有些音色是坏掉的

</br>

_提取音色_

[参考代码](https://huggingface.co/mechanicalsea/speecht5-vc/blob/main/manifest/utils/prep_cmu_arctic_spkemb.py)


精简后：

```python

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device}, savedir="pretrained_models/spkrec-xvect-voxceleb")

def make_npy(role, normalize = True):
    signal, fs =torchaudio.load(f'sample_audio/{role}.wav')

    print(f"Audio sample: {fs}")
    if fs != 16000:
        print(f"Resample: {fs} -> 16000")
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)
        # display(Audio(signal, rate=16000))
    
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        if normalize:
            embeddings = F.normalize(embeddings, dim=2)
        
        # 双通道时，只取第一个
        if embeddings.shape[0] == 2:
            embeddings = embeddings.squeeze().cpu().numpy()[0]
        else:
            embeddings = embeddings.squeeze().cpu().numpy()

    # print(embeddings)
    np.save(f"./spkemb/{role}.npy", np.array(embeddings))
    print(f"Saved ./spkemb/{role}.npy success! {embeddings.shape}")
```

> 使用以上的代码，可以对任意wav提取音色；
> 相关建议：
> (1)原音频不要混响、要清澈人声。
> (2)长度没要求，一般 30s
> (3)最终质量比较玄学，与音频长度、音频质量、是否降噪、是否调音有一定关系



_对比embedding_


```python
## 对比 embedding
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))

x = [i for i in range(1, 513)]

plt.plot(x, np.load("spkemb/speaker1.npy"))
plt.plot(x, np.load("spkemb/speaker2.npy"))
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230330232325.png" style ="zoom:80%">

如图，相同角色不同片段的 embedding 几乎一致；


做这个主要是看取相同角色 embedding 的平均值是否有用。
结论：用处不多。


## 最终代码

```python
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from IPython.display import Audio
import torch.nn.functional as F
import torch
import soundfile as sf
import gradio as gr
import numpy as np
import os


# Speech T5
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Speechbrain : to extract speaker embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device}, savedir="pretrained_models/spkrec-xvect-voxceleb")


show_text = """ hello, world!
"""


speaker_embeddings = {
    "BDL": "spkemb/cmu_us_bdl_arctic-wav-arctic_a0009.npy",
    "CLB": "spkemb/cmu_us_clb_arctic-wav-arctic_a0144.npy",
    "SLT": "spkemb/cmu_us_slt_arctic-wav-arctic_a0508.npy"
}


def make_npy(role, normalize = True):
    signal, fs = torchaudio.load(f'sample_audio/{role}.wav')
    
    print(f"Audio sample: {fs}")
    if fs != 16000:
        print(f"Resample: {fs} -> 16000")
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)
        display(Audio(signal, rate=16000))
    
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        if normalize:
            embeddings = F.normalize(embeddings, dim=2)
        
        # 双通道时，只取第一个
        if embeddings.shape[0] == 2:
            embeddings = embeddings.squeeze().cpu().numpy()[0]
        else:
            embeddings = embeddings.squeeze().cpu().numpy()

    # print(embeddings)
    np.save(f"./spkemb/{role}.npy", np.array(embeddings))
    print(f"Saved ./spkemb/{role}.npy success! {embeddings.shape}")
    

def merge_npy(role1, role2, role3):
    a = np.load(f"./spkemb/{role1}.npy")
    b = np.load(f"./spkemb/{role2}.npy")
    np.save(f"./spkemb/{role3}.npy", (a+b)/2)
    

def test_npy(role, text = show_text):
    print(f"Load speaker embedding from ./spkemb/{role}.npy ")
    a = np.load(f"./spkemb/{role}.npy")
    print(a)
    speaker_embedding = torch.tensor(a).unsqueeze(0)
    
    inputs = processor(text = text, return_tensors="pt")

    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    sf.write("speech.wav", speech.numpy(), samplerate=16000)
    display(Audio('speech.wav'))


def show_base():

    def tts_fn(text, speaker):
        inputs = processor(text=text, return_tensors="pt")

        a = np.load(speaker_embeddings[speaker[:3]])
        speaker_embedding = torch.tensor(a).unsqueeze(0)

        # speaker_embedding = np.load(speaker_embeddings[speaker[:3]])
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

        sf.write("speech.wav", speech.numpy(), samplerate=16000)

        # data, samplerate = sf.read("speech.wav")
        out_path = "speech.wav"
        # print(type(data))
        # data is numpy.ndarray
        return "success", out_path   

    app = gr.Blocks()
    with app:
        with gr.Tab("Speech T5 Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label = "Text",
                                          placeholder = "Type your sentence here",
                                          value = show_text,
                                          elem_id = "tts-input")

                    speaker = gr.Radio(label="Speaker", choices=["BDL (male)", 
                                                                 "CLB (female)",
                                                                 "SLT (female)"], value="CLB (female)")
                    
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn,
                              inputs=[textbox, speaker],
                              outputs=[text_output, audio_output])

    app.launch() 
```


## 其他说明


公共问题：
- 只能16000
- 很容易电音
- 数字无法正确 TTS
- 有的时候生成的会跳词，有时不会
- 连词符号 - , 停顿异常
- 生成速度偏慢，平均1s 音频用 1s 生成
- 微调比较困难



!> 建议：还是去用 VITS 吧