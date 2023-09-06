

[Real-Time-Voice-Cloning 码仓地址](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

[演示视频](https://www.bilibili.com/video/BV1Va4y1j7ur/)



模型分为三部分：
- encoder.pt 17mb
- vocoder.pt 52mb
- synthesizer.pt 354mb


</br>


简单测试：
```python
import soundfile as sf
from IPython.display import Audio
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa


# 运行 demo_cli 会自动下载模型
encoder_weights = Path("saved_models/default/encoder.pt")
vocoder_weights = Path("saved_models/default/vocoder.pt")
syn_dir = Path("saved_models/default/synthesizer.pt")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

file1 = "/home/ec2-user/SageMaker/Real-Time-Voice-Cloning/denoise_audio/sophia-normal2.wav"

def synth(text):
    in_fpath = Path(file1)
    reprocessed_wav = encoder.preprocess_wav(in_fpath)
    
    original_wav, sampling_rate = librosa.load(in_fpath)
    
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    
    embed = encoder.embed_utterance(preprocessed_wav)

    #embed, sampling_rate 每个角色唯一
    print("Synthesizing new audio...")
    
    #need list
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    spec = specs[0]
    
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    
    # Trim excess silences to compensate for gaps in spectrograms
    generated_wav = encoder.preprocess_wav(generated_wav)

    display(Audio(generated_wav, rate=synthesizer.sample_rate))

display(Audio(file1, rate=synthesizer.sample_rate))
synth(text1)
```

>toolbox 跑通，效果不好；不仅杂音，还部分跳词，有的声音在蒙混过关。

Welcome to the toolbox! To begin, load an utterance from your datasets or record one yourself.

Once its embedding has been created, you can synthesize any text written here.

The synthesizer expects to generate outputs that are somewhere between 5 and 12 seconds.

To mark breaks, write a new line. Each line will be treated separately.

Then, they are joined together to make the final spectrogram. Use the vocoder to generate audio.

The vocoder generates almost in constant time, so it will be more time efficient for longer inputs like this one.

On the left you have the embedding projections. Load or record more utterances to see them.

If you have at least 2 or 3 utterances from a same speaker, a cluster should form.

Synthesized utterances are of the same color as the speaker whose voice was used, but they're represented with a cross.

