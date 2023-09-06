
```python
signal, fs =torchaudio.load(f'sample_audio/{role}.wav')


# 重采样
# 或者使用 librosa
print(f"Audio sample: {fs}")
if fs != 16000:
    print(f"Resample: {fs} -> 16000")
    signal = torchaudio.transforms.Resample(fs, 16000)(signal)
```

```python
signal, fs = torchaudio.load(file)
display(Audio(signal, rate = 48000))

# 如果采样率高转低，时长变成，声音变低沉
```
