
展示音频 demo

```python
from IPython.display import Audio

display(Audio('/content/speech.wav'))

display(Audio(signal, rate = 22050))
# rate must be specified when data is a numpy array or list of audio samples
```

