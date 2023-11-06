
nsf hifigan

contentvec



- [x] sovits 流程跑通
- [ ] tensorboard 使用
- [ ] hugg 两个库
- [ ] 找音频训练
- [ ] transformer att
- [ ] fork svc 并修改
- [ ] 文件结构， vits 改为子级，再加上 edgetts

https://github.com/PlayVoice/so-vits-svc-5.0 两份代码差异好大

https://github.com/MaxMax2016

https://huggingface.co/docs/transformers/model_doc/bark

https://huggingface.co/docs/transformers/model_doc/vits

diffusion audio 效果？

语音相关：
- [ ] 多语种 tts 部分
- [ ] svc 找语音新音色
- [ ] 界面支持对话
- [ ] 源码整合改进


- [ ] 课程笔记整理
- [ ] 更新 develop
- [ ] 系统论的书
- [ ] MHA
- [ ] chatglm3
- [ ] vgg 代码写法，看起来  vits 代码不够优雅
- [ ] 多读 torch 文档



日志报错：Segmentation fault，然后服务直接挂掉了，弄个定时任务告警


https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master


https://github.com/ultralytics/yolov5/issues/8914 这外国人这么好，，，


----------

vits 是对 cuda 等版本无要求的

封装有些多层


vits 报错

估计版本问题，TypeError: mel() takes 0 positional arguments but 5 were given

mel = librosa_mel_fn(sr = sampling_rate, n_fft = n_fft, n_mels = num_mels, fmin = fmin, fmax = fmax)

---------

File "/home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/serialization.py", line 291, in __exit__
self.file_like.write_end_of_file()
RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos 320 vs 240

晕了，这个的原因是存储空间不足

--------

UserWarning: stft with return_complex=False is deprecated. In a future pytorch release, stft will return complex tensors for all inputs, and return_complex=False will raise an error.
Note: you can still call torch.view_as_real on the complex output to recover the old return format. (Triggered internally at /opt/conda/conda-bld/pytorch_1686274778240/work/aten/src/ATen/native/SpectralOps.cpp:862.)
  return _VF.stft(input, n_fft, hop_length, win_length, window,  # type: ignore[attr-defined]

这个详细信息在 librosa 的 mel spectrogram 中，被封起来了

--------

  File "/home/ec2-user/anaconda3/envs/pytorch_p310/lib/python3.10/site-packages/torch/serialization.py", line 283, in __init__
    super().__init__(torch._C.PyTorchFileReader(name_or_buffer))
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory

重新弄

---------

重新弄之后，新的报错


ft
    return _VF.stft(input, n_fft, hop_length, win_length, window,  # type: ignore[attr-defined]
RuntimeError: stft requires the return_complex parameter be given for real inputs, and will further require that return_complex=True in a future PyTorch release.



output = torch.stft(input, n_fft=2048, return_complex=True)

----------

修复之后：RuntimeError: mat1 and mat2 shapes cannot be multiplied (80x513 and 64x513)

报错报晕了，还是按原本 requirements 跑一遍

colab 都跑不了这个,

使用 return_complex=False 解决，原本是因为 return_complex=True ,,,

之后还是这个报错：RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory

------------

读 spec 文件的时候发生报错，难道是第一个报错修复引起的？

librosa_mel_fn(sr = sampling_rate, n_fft = n_fft, n_mels = num_mels, fmin = fmin, fmax = fmax)

#就是这里出的错，指向这个： DUMMY1/LJ018-0084.spec.pt
#MALE 
#估计是文件循坏什么的，其它训练数据好好的

-----------

重新弄数据集，其他：中途语法错误，爆显存

终于通了!!!

这下载数据解压还能错误，1% 的运气？


--------

倒霉，今天的 sovitssvc 又是报错的一天。

已经解决三四个了，现在这个错有点意思，记录一下

RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`

可能原因1：爆显存，修改configs 中 diffusion.yaml batch_size，应该不是，显存才到 1800

多运行几遍，是个新的报错：

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ec2-user/SageMaker/so-vits-svc/preprocess_hubert_f0.py", line 172, in <module>
    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device)
  File "/home/ec2-user/SageMaker/so-vits-svc/preprocess_hubert_f0.py", line 128, in parallel_process
    task.result()
  File "/home/ec2-user/anaconda3/envs/python3/lib/python3.10/concurrent/futures/_base.py", line 458, in result
    return self.__get_result()
  File "/home/ec2-user/anaconda3/envs/python3/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

从后往前导不出来，看代码哪里可能出错，

定位到 c = hmodel.encoder(wav16k) 这行代码不能正确运行，不会又是模型下载损坏？

用的是 GPU

speech_encoder = hps["model"]["speech_encoder"]
hmodel = utils.get_speech_encoder(speech_encoder, device=device)
c = hmodel.encoder(wav16k)

多进程问题？？？

有的不会报错，运行几条才报错

-------

修改为单进程之后报错换了个位置，同样的报错，，，

去掉 cpu(), 去掉 tqdm

。。。弄不好

------------

Day2 CUDA_LAUNCH_BLOCKING=1

还是这个报错

  File "/home/ec2-user/SageMaker/so-vits-svc/preprocess_hubert_f0.py", line 172, in <module>
    parallel_process(filenames, num_processes, f0p, args.use_diff, mel_extractor, device)
  File "/home/ec2-user/SageMaker/so-vits-svc/preprocess_hubert_f0.py", line 128, in parallel_process
    task.result()
  File "/home/ec2-user/anaconda3/envs/python3/lib/python3.10/concurrent/futures/_base.py", line 458, in result
    return self.__get_result()
  File "/home/ec2-user/anaconda3/envs/python3/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`


不至于啊，只是个数据预处理救这么个错，，

将进程切换到 8 报错：

concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

UserWarning: resource_tracker: There appear to be 16 leaked semaphore objects to clean up at shutdown

某个进程报错，还有 16个未信号量

切换到单进程，使用 top 看看是不是资源的问题

会占用满 CPU， 估计是  tensor 运算放到 CPU 导致的

RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`


----------

检查到 cudnn 不兼容：

RuntimeError: cuDNN version incompatibility: PyTorch was compiled  against (8, 9, 2) but found runtime version (8, 7, 0). PyTorch already comes bundled with cuDNN. One option to resolving this error is to ensure PyTorch can find the bundled cuDNN.Looks like your LD_LIBRARY_PATH contains incompatible version of cudnnPlease either remove it from the path or install cudnn (8, 9, 2)

回退 torch 至 2.0.1

用的版本
2.0.1
11.8
8700

还是占着 CPU 在 跑；

按照它的环境装一遍，内部 torch cuda cudnn 是兼容的，就解决了

今天的可能原因：装的 torch 版本与 预置的 cudnn 不兼容

一定预装 require









holocubic, 作用：显示时钟，图片，温度信息

ML/DL/ROADMAP

huawei ai tutorial

https://github.com/ZachGoldberg/Startup-CTO-Handbook/blob/main/StartupCTOHandbook.md#speed-is-your-friend

https://github.com/ByteByteGoHq/system-design-101

- [ ] blog ML/DL/transformer 整理
- [ ] 整理归档，ASR 课程笔记

https://www.fast.ai/


[深度学习之图像翻译与风格化GAN-理论与实践](https://www.bilibili.com/video/BV1Wr4y1b77B)

bili 论文解读

- electron开发，代替 TODO，自定义背景，等


- [ ] 右侧添加一个单页面的 目录
- [ ] 整理 torch 文档到 blog


前端应用程序开发：https://github.com/Moonvy/OpenPromptStudio

https://github.com/wangshub


https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3

http://nlp.seas.harvard.edu/2018/04/03/attention.html


https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

https://www.cloudskillsboost.google/journeys/118
