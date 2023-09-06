
**pydub 进行格式转换**

```python
from pydub import AudioSegment
import os

raw_audio_dir = "./raw_audio/"

def mp32wav(filepath):
    filelist = list(os.walk(raw_audio_dir))[0][2]
    for file in filelist:
        print(raw_audio_dir+file)
        song = AudioSegment.from_mp3(raw_audio_dir+file)
        song.export("./PURE/" + file[:-4] + ".wav", format="wav")
    
wav = mp32wav("./raw_audio")
```