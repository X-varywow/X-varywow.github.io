


>读取音频

```python
import librosa
from IPython.display import Audio

librosa.get_samplerate("speech.wav")
# -> 16000


# rate, default = 22050
# mono = True, 是否将信号转换为单声道
wave, sr = librosa.load("speech.wav", sr=16000)

display(Audio("speech.wav", rate=16000))
print(wave, len(wave), sr)

```

如何写入音频，参考模块 soundfile

>波形图

```python
%matplotlib inline

import matplotlib.pyplot as plt
import librosa.display

wave, sr = librosa.load("speech.wav")
display(Audio("speech.wav"))

# use figsize to indicate width, height
plt.figure(figsize=(20, 5))
librosa.display.waveplot(wave, sr=sr)
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230329235353.png">

>频谱图 Spectrogram

很常用的一个特征;

```python
%matplotlib inline

import matplotlib.pyplot as plt
import librosa.display

wave, sr = librosa.load("speech.wav")
display(Audio("speech.wav"))

# use figsize to indicate width, height
plt.figure(figsize=(20, 5))

wave = librosa.stft(wave)
wave = librosa.amplitude_to_db(abs(wave))

# y_axis: log, hz
librosa.display.specshow(wave, sr=sr, x_axis="time", y_axis="hz")
plt.colorbar()
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230329235354.png">

> 梅尔频率倒谱系数（MFCC）

```python
%matplotlib inline

import matplotlib.pyplot as plt
import librosa.display

wave, sr = librosa.load("speech.wav")
display(Audio("speech.wav"))

# use figsize to indicate width, height
plt.figure(figsize=(20, 5))

mfccs = librosa.feature.mfcc(y=wave, sr=sr)
wave = librosa.amplitude_to_db(abs(wave))

# y_axis: log, hz
librosa.display.specshow(mfccs, sr=sr, x_axis="time", y_axis="hz")
plt.colorbar()
```

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230329235355.png">


>添加白噪声

```python
import librosa
import numpy as np
import soundfile as sf

# 添加噪声时保留0值
def add_noise(audio_path,percent=0.02, sr = 16000):
    src, sr = librosa.load(audio_path, sr = sr)
    # print(min(src), max(src), len([i for i in src if i==0.0]))
    random_values = np.random.rand(len(src))
    # print(random_values, len(src))
    src = np.where(src!=0.0, src + percent*random_values, 0.0)
    sf.write("speech_v2.wav", src, sr)
    
add_noise("speech.wav")
display(Audio('speech_v2.wav'))

# 根据 len(src) 和 sr 可以得出声音时长
```

> Advanced usage

```python
# Feature extraction example
import numpy as np
import librosa

# Load the example clip
y, sr = librosa.load(librosa.ex('nutcracker'))

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                             sr=sr)

# Compute MFCC features from the raw signal
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

# And the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                        sr=sr)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram,
                                beat_frames,
                                aggregate=np.median)

# Finally, stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
```

> Vocal separation

https://librosa.org/doc/latest/auto_examples/plot_vocal_separation.html


-------------

参考资料：

[Quickstart](https://librosa.org/doc/latest/tutorial.html#quickstart)

[音频处理库—librosa的安装与使用](https://blog.csdn.net/zzc15806/article/details/79603994)

[librosa音频处理教程](https://developer.aliyun.com/article/932895)

[librosa语音信号处理](https://www.jianshu.com/p/8d6ffe6e10b9?spm=a2c6h.12873639.article-detail.146.3f3b4c67w9UBgx)

https://www.cnblogs.com/LXP-Never/p/11561355.html

[github 地址](https://github.com/librosa/librosa)，5.8k

