
[openai whisper 文章](https://openai.com/research/whisper)

[官方仓库](https://github.com/openai/whisper)，31.5k。Robust Speech Recognition via Large-Scale Weak Supervision

[hugging face top1 wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english)


>使用 whisper

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

https://github.com/k2-fsa/icefall