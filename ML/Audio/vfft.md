
[colab demo](https://colab.research.google.com/drive/1pn1xnFfdLK63gVXDwV4zCXfVeo8c-I-0)

## Preface

- 模型相关
  - `finetune_speaker_v2`
- 工具相关
  - `video2audio`
    - moviepy.editor.AudioFileClip
  - `denoise_audio`
    - 使用 Demucs 分离人声
    - https://zhuanlan.zhihu.com/p/510755328
  - `long_audio_transcribe`
    - 使用 openai-whisper 进行 ASR 和 分割、标注
    - https://github.com/openai/whisper
  - `short_audio_transcribe`



## finetune


preprocess_v2.py 整理处理好的文件，训练信息 "long_character_anno.txt"

torch.multiprocessing.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


这个分桶是？？？

```python
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
```


预训练模型是 huggingfacce 上的  pth 文件


```python
%run preprocess_v2.py --add_auxiliary_data True --languages

%reload_ext tensorboard
%tensorboard --logdir "./OUTPUT_MODEL"
Maximum_epochs = "40" #@param [20, 30, 40, 50, 60]
!python finetune_speaker_v2.py -m "./OUTPUT_MODEL" --max_epochs "{Maximum_epochs}" --drop_speaker_embed True
```

preprocess_v2.py

new_annos,


## other

whisper  达到asr商业标准，

内置了现实和原神等二次元语音的数据，数据不足时也可 ADD_AUXILIARY。

生成带有二次元音色的语音


`已跑通`，音色还是可以的，音色太二次元了，停顿不太正常（可能日语导致的）。


[MoeGoe](https://github.com/CjangCjengh/MoeGoe)