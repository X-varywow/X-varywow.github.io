
使用自带的模型，可以实现一个几乎完美效果的语音克隆。（女声，应该是 ljs 数据训练出来的）

但是没有开源

```python
import torch, torchaudio
import requests
import IPython.display as display


# Download the HuBERT content encoder (either hubert_soft or hubert_discrete):
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()
# Download the acoustic model (either hubert_soft or hubert_discrete)
acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()
# Download the vocoder (either hifigan_hubert_soft or hifigan_hubert_discrete)
hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").cuda()
```

```python
with open("example.wav", "wb") as file:
  response = requests.get("https://drive.google.com/uc?export=preview&id=1Y3KuPAhB5VcsmIaokBVKu3LUEZOfhSu8")
  file.write(response.content)

# from google.colab import files
# uploaded = files.upload()
```

```python
source, sr = torchaudio.load("RY0001-0082.wav")
source = torchaudio.functional.resample(source, sr, 16000)
source = source.unsqueeze(0).cuda()

with torch.inference_mode():
    # Extract speech units
    units = hubert.units(source)
    # Generate target spectrogram
    mel = acoustic.generate(units).transpose(1, 2)
    # Generate audio waveform
    target = hifigan(mel)

display.Audio(source.squeeze().cpu(), rate=16000)
```

```python
display.Audio(target.squeeze().cpu(), rate=16000)
```

------------

参考资料：
- [svc repo](https://github.com/bshall/soft-vc)
- [svc demo](https://bshall.github.io/soft-vc/)