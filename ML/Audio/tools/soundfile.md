
[官方文档](https://pysoundfile.readthedocs.io/en/latest/)

>简单读写

```python
import soundfile as sf

data, samplerate = sf.read('existing_file.wav')
sf.write('new_file.flac', data, samplerate)
```

>写


```python
import numpy as np
import soundfile as sf

rate = 44100
data = np.random.uniform(-1, 1, size=(rate * 10, 2))

# Write out audio as 24bit PCM WAV
sf.write('stereo_file.wav', data, samplerate, subtype='PCM_24')

# Write out audio as 24bit Flac
sf.write('stereo_file.flac', data, samplerate, subtype='PCM_24')

# Write out audio as 16bit OGG
sf.write('stereo_file.ogg', data, samplerate, subtype='vorbis')
```