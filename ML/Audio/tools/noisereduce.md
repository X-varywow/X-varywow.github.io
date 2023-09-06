
```python
from scipy.io import wavfile
import noisereduce as nr
from IPython.display import Audio

# load data
rate, data = wavfile.read("before.wav")
print(rate, data)
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("after_reduce_noise.wav", rate, reduced_noise)


display(Audio("before.wav", rate=16000))
display(Audio("after_reduce_noise.wav", rate=16000))
```