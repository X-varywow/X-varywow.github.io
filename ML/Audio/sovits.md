
## _preface_

An implementation of the combination of Soft-VC and VITS

码仓：[svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)


[sovits4.0一键训练/推理脚本.ipynb](https://colab.research.google.com/drive/1hGt9XowC07NGmXxKNJvY5N64uMdd435M)


https://www.bilibili.com/video/BV1H24y187Ko/ (纯工程应用向，实用性强)， [视频教程2](https://www.bilibili.com/video/BV1iL411y7Z5/)

[svc-develop-team colab demo1](https://colab.research.google.com/github/svc-develop-team/so-vits-svc/blob/4.0/sovits4_for_colab.ipynb) 

[colab demo2](https://colab.research.google.com/github/34j/so-vits-svc-fork/blob/main/notebooks/so-vits-svc-fork-4.0.ipynb)

[soft_vc_demo3](https://colab.research.google.com/github/bshall/soft-vc/blob/main/soft-vc-demo.ipynb)

[colab demo1](https://colab.research.google.com/drive/1WjOZFyJVdP7-uozL5TKDekqnSB4rmMm-)

实用性及效果很强；[AI 孙燕姿](https://www.bilibili.com/video/BV1Rm4y187N1/) 用的也是这个模型。

试了一下，10分钟高质量语音就能出来一个较好的音色；社区与文档也好。



</br>

## _训练流程_

（1）环境准备

```bash
cd so-vits-svc

source activate
conda activate pytorch_p310
pip install --upgrade pip setuptools numpy numba
pip install -r requirements.txt

# 可选
sudo yum install espeak -y
ipython kernel install --user --name svc
```

（2）原始数据准备

```
raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

若数据为长音频，推荐使用以下工具进行处理:

- UVR5来分离人声和伴奏，去混响
- 使用 audio-slicer 将长音频切片，参考： [官方 repo](https://github.com/openvpi/audio-slicer)




（3）处理原始数据

```bash
# dataset_raw -> dataset
python resample.py

# flist config
# vol_aug，响度嵌入，使声音的响度依据输入而不依赖训练数据。
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug

# hubert 特征
python preprocess_hubert_f0.py --use_diff --num_processes 8
```

之后在 dataset/44k 文件夹下，


一个音频8个文件：
- wav
- spec.pt
- aug_mel
- aug_vol
- f0
- mel
- soft
- vol


（4）下载预训练模型

--1

```bash
# contentvec
wget -P pretrain/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O checkpoint_best_legacy_500.pt
# Alternatively, you can manually download and place it in the hubert directory
```

--2

之后的默认参数 hubert_f0 时也需要下个东西：

```bash
wget https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip
unzip rmvpe.zip 
mv model.pt ./pretrain/rmvpe.pt
```


--3

预训练底模下载

预训练底模文件： G_0.pth D_0.pth 放在logs/44k目录下

扩散模型预训练底模文件： model_0.pt 放在logs/44k/diffusion目录下


```bash
wget https://huggingface.co/datasets/ms903/Diff-SVC-refactor-pre-trained-model/resolve/main/Diffusion-SVC/shallow_512_30/model_0.pt

mv model_0.pt ./logs/44k/diffusion/
```

或者去：

https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/tree/main

--4

使用扩散会需要 nsf_hifigan

```bash
wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip nsf_hifigan_20221211.zip
```





（5）开始训练

使用的 A10G 24GB 显卡，将 batch 修改为 20

```bash
python train.py -c configs/config.json -m 44k

# （可选）扩散模型
python train_diff.py -c configs/diffusion.yaml
```







## _模型说明_

参考：https://zhuanlan.zhihu.com/p/630115251

encoder 特征编码器

vocoder 声码器




## _speech encoder_

- contentvec (recommended)
  - vec768l12
    - 4.1 版本默认（能直观感受到训练速度慢了很多）
  - vec256l9
    - 应该是 4.0 版本默认的编码器（换到 4.1 需要修改 config）
- hubertsoft
- whisper-ppg
- cnhubertlarge
- dphubert
- whisper-ppg-large
- wavlmbase+


负责提取特征， contentvec 感觉只是个普通的文本表示模型，甚至不如 word2vec

------------

ContentVec模型在语音合成中的应用主要是用于文本到语音的转换过程中，将输入的文本转化为对应的语音信号。

在语音合成中，ContentVec模型首先将输入的文本进行分词和词性标注，然后根据每个词的词频或TF-IDF权重，构建一个词向量。这些词向量可以表示文本的内容信息。接下来，将这些词向量输入到语音合成模型中，通过模型的训练和优化，生成对应的语音信号。

与ContentVec模型相比，Hubert模型是一种基于Transformer架构的语音合成模型。它不仅可以将文本转化为语音信号，还可以直接从原始音频中提取语音特征，并进行语音合成。Hubert模型通过自监督学习的方式，学习到语音和文本之间的对应关系，从而实现文本到语音的转换。

相比之下，ContentVec模型更注重于文本的内容表示，而Hubert模型更注重于语音特征的提取和语音合成。Hubert模型可以更准确地捕捉语音的语调、音色等特征，从而生成更自然流畅的语音。而ContentVec模型则更适用于处理大量的文本数据，用于文本的内容表示和分析。


--------

NSF HiFiGAN是一种基于神经网络的音频合成模型，用于生成高保真的人类语音

基于 GANs



-------

f0_predictor has the following options

crepe
dio
pm
harvest
rmvpe
fcpe



1. 支持响度嵌入

preprocess_flist_config 时 vol_aug，使声音的响度依据输入而不依赖训练数据。 是需要的;

2. 引入浅扩散机制

将原始输出音频转为 mel谱图，显著改善电音、底噪等问题



## 文件结构


dataset_raw  原始音频

dataset/44k 用于训练


wav_path = str(Path(raw_audio_path).with_suffix('.wav'))

音频要求：44100Hz and mono

checkpoint_best_legacy_500.pt 默认的 hubert 模型，提取音频的特征。

#### resample.py

```python
# 多进程 & tqdm

from multiprocessing import Pool, cpu_count
processs = 30 if cpu_count() > 60 else (cpu_count()-2 if cpu_count() > 4 else 1)
pool = Pool(processes=processs)

for _ in tqdm(pool.imap_unordered(process, [(spk_dir, i, args) for i in os.listdir(spk_dir) if i.endswith("wav")])):
    pass
```

#### preprocess_flist_config.py

```python
config_template = json.load()

rb 读取文件，.getnframes()

# val 只有两条？
shuffle(wavs)
train += wavs[2:]
val += wavs[:2]

config_template["spk"] = spk_dict

# make config
with open("configs/config.json", "w") as f:
    json.dump(config_template, f, indent=2)
```

#### preprocess_hubert_f0.py

dataset_raw 可以删除了，wav 在 44k, filelist,config 也都生成了

主要三部分：

（1）转 16k， get_hubert_content() -> .soft.pt

```python
with torch.no_grad():
    logits = hmodel.extract_features(**inputs)
    feats = hmodel.final_proj(logits[0])
return feats.transpose(1, 2)
```

（2）不转，compute_f0_dio() -> .f0.npy

```python
def compute_f0_dio(wav_numpy, p_len=None, sampling_rate=44100, hop_length=512):
    import pyworld
    if p_len is None:
        p_len = wav_numpy.shape[0]//hop_length
    f0, t = pyworld.dio(
        wav_numpy.astype(np.double),
        fs=sampling_rate,
        f0_ceil=800,
        frame_period=1000 * hop_length / sampling_rate,
    )
    f0 = pyworld.stonemask(wav_numpy.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    return resize_f0(f0, p_len)
```

（3）Process spectrogram: spec_path = filename.replace(".wav", ".spec.pt")

## 参数说明

> 常用推理：-n {RAW2} -m {MODEL} -c {CONFIG} -s {SPEAKER} -a -fmp


- 基本参数
  - `-m` model_path
  - `-c` config_path
  - `-s` speaker
  - `-n` clean wav be cloned
  - `-t` pitch adjustment，男女声转换要用
- 其他参数
  - `-a` Automatic pitch prediction for voice conversion.
  - `-lg` 平滑过渡
  - `-cm` cluster_model_path
  - `-cr`
  - `-fmp` f0均值滤波，改善哑音
  - `-eh`, nsf_hifigan enhaner，可以改善训练很少的模型，但对好模型负向


聚类模型的作用：The clustering scheme can reduce timbre leakage and make the trained model sound more like the target's timbre (although this effect is not very obvious), but using clustering alone will lower the model's clarity (the model may sound unclear). Therefore, this model adopts a fusion method to linearly control the proportion of clustering and non-clustering schemes. In other words, you can manually adjust the ratio between "sounding like the target's timbre" and "being clear and articulate" to find a suitable trade-off point.


**Automatic f0 prediction**, which can be used for automatic pitch prediction during voice conversion.


**聚类方案** 可以减少音色泄漏，使训练后的模型听起来更像目标的音色（虽然这种效果不是很明显），但单独使用聚类会降低模型的清晰度（模型听起来可能不清晰）。因此，该模型采用融合的方法，线性控制聚类和非聚类方案的比例。换句话说，你可以手动调整“听起来像目标音色”和“清晰、口齿清晰”之间的比例，找到一个合适的权衡点。


**F0均值滤波** 可以有效降低预测的音高波动引起的嘶哑（混响或和声引起的嘶哑暂时无法消除）。此功能在某些歌曲上得到了很大改进。但是，有些歌曲走调了。如果推理后歌曲出现哑巴，可以考虑打开。

```python
# 一次完整参数的 infer
wav_filename = "YourWAVFile.wav"  #@param {type:"string"}
model_filename = "G_210000.pth"  #@param {type:"string"}
model_path = "/content/so-vits-svc/logs/44k/" + model_filename
speaker = "YourSpeaker"  #@param {type:"string"}
trans = "0"  #@param {type:"string"}
cluster_infer_ratio = "0"  #@param {type:"string"}
auto_predict_f0 = False  #@param {type:"boolean"}
apf = ""
if auto_predict_f0:
  apf = " -a "
f0_mean_pooling = False  #@param {type:"boolean"}
fmp = ""
if f0_mean_pooling:
  fmp = " -fmp "
enhance = False  #@param {type:"boolean"}
ehc = ""
if enhance:
  ehc = " -eh "
#@markdown

#@markdown Generally keep default:
config_filename = "config.json"  #@param {type:"string"}
config_path = "/content/so-vits-svc/configs/" + config_filename
kmeans_filenname = "kmeans_10000.pt"  #@param {type:"string"}
kmeans_path = "/content/so-vits-svc/logs/44k/" + kmeans_filenname
slice_db = "-40"  #@param {type:"string"}
wav_format = "flac"  #@param {type:"string"}
wav_output = "/content/so-vits-svc/results/" + wav_filename + "_" + trans + "key" + "_" + speaker + "." + wav_format

%cd /content/so-vits-svc
!python inference_main.py -n {wav_filename} -m {model_path} -s {speaker} -t {trans} -cr {cluster_infer_ratio} -c {config_path} -cm {kmeans_path} -sd {slice_db} -wf {wav_format} {apf} {fmp} {ehc}

#@markdown

#@markdown If you dont want to download from here, uncheck this.
download_after_inference = True  #@param {type:"boolean"}

if download_after_inference:
  from google.colab import files
  files.download(wav_output)
```

## 推理

```shell
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -s "nen" -n "君の知らない物語-src.wav" -t 0
```

```python
chunks = slicer.cut(wav_path, db_thresh=slice_db)

audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

def infer(self, speaker, tran, raw_path,
              cluster_infer_ratio=0,
              auto_predict_f0=False,
              noice_scale=0.4,
              f0_filter=False,
              F0_mean_pooling=False,
              enhancer_adaptive_key = 0,
              cr_threshold = 0.05
              ):

        speaker_id = self.spk2id.__dict__.get(speaker)
        if not speaker_id and type(speaker) is int:
            if len(self.spk2id.__dict__) >= speaker:
                speaker_id = speaker
        sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)
        c, f0, uv = self.get_unit_f0(raw_path, tran, cluster_infer_ratio, speaker, f0_filter,F0_mean_pooling,cr_threshold=cr_threshold)
        if "half" in self.net_g_path and torch.cuda.is_available():
            c = c.half()
        with torch.no_grad():
            start = time.time()
            audio = self.net_g_ms.infer(c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0, noice_scale=noice_scale)[0,0].data.float()
            if self.nsf_hifigan_enhance:
                audio, _ = self.enhancer.enhance(
                                                                        audio[None,:], 
                                                                        self.target_sample, 
                                                                        f0[:,:,None], 
                                                                        self.hps_ms.data.hop_length, 
                                                                        adaptive_key = enhancer_adaptive_key)
            use_time = time.time() - start
            print("vits use time:{}".format(use_time))
        return audio, audio.shape[-1]

```

打开sagemaker -> 部署服务:

```bash
source activate python3
cd ~/SageMaker/vitssvc

pip install --upgrade pip setuptools numba
pip install -r requirements.txt

sudo yum install espeak -y
ipython kernel install --user --name vitssvc

cd fairseq
pip install --editable ./

cd ..
python zz_webui.py
```


## other 

[diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)

应该是后端部分，mel 频谱特征使用 diffusion 的方式扩散生成



### 6.1 复用模型

[hugging face models](https://huggingface.co/models?search=so-vits-svc-4.0)


19 个 较好的模型：
- https://huggingface.co/xgdhdh/so-vits-svc-4.0/tree/main （caster）(morgan)(saber)
- https://huggingface.co/therealvul/so-vits-svc-4.0/tree/main (很多)(含 kmeans)（done）
- https://huggingface.co/TachibanaKimika/so-vits-svc-4.0-models/tree/main (kiriga)*10
- https://huggingface.co/melicat/so-vits-svc-4.0/tree/main （chenzhuoxuan）
- https://huggingface.co/marcoc2/so-vits-svc-4.0-models/tree/main （gaga）*10
- https://huggingface.co/Nardicality/so-vits-svc-4.0-models/tree/main (biden)*3
- https://huggingface.co/RAYTRAC3R/so-vits-svc-4.0/tree/main



### 6.2 keep_ckpts 实现

```python
def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
  """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
  ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
  name_key = (lambda _f: int(re.compile('._(\d+)\.pth').match(_f).group(1)))
  time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
  sort_key = time_key if sort_by_time else name_key
  x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pth')], key=sort_key)
  to_del = [os.path.join(path_to_models, fn) for fn in
            (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
  del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
  del_routine = lambda x: [os.remove(x), del_info(x)]
  rs = [del_routine(fn) for fn in to_del] 
```



### 6.3 对比 vits

model 是变了的

infer 多了些参数

_infer 对比_

多了些参数，模型层面改动了

```python
## vits

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()

net_g.eval()

utils.load_checkpoint("/path/to/pretrained_ljs.pth", net_g, None)

net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
```


```python
## svc infer

svc_model.infer(spk, tran, raw_path,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                F0_mean_pooling = F0_mean_pooling,
                enhancer_adaptive_key = enhancer_adaptive_key,
                cr_threshold = cr_threshold
                )
```

```python
## svc infer_tool.py

class svc:
    with torch.no_grad():
        audio = self.net_g_ms.infer(c, f0=f0, g=sid, uv=uv, predict_f0=auto_predict_f0, noice_scale=noice_scale)[0,0].data.float()
        if self.nsf_hifigan_enhance:
                audio, _ = self.enhancer.enhance
```

清楚了些，除了本身的 vits+ softvc 模型，还有 hubert 用于提取特征， cluster 用于减少音色泄露， enhancer。


### Tricks

#### 数据方面

数据尽量不要带着音响效果

语音尽量去除底噪和混响

音频质量＞音频数量


sovits 推荐干声 1-2h，结果影响：数据集质量，轮次，数量;


#### 参数方面


未添加聚类，聚类达到效果？

效果： 2h 可达成目标音色。


视频1 

混响想过去干净，数据集质量要求；

去掉和声

自动 f0 预测，语音时自动变调



-----------

参考资料：
- https://github.com/bshall/soft-vc
- chat-gpt
