
## VToonify 简介

> state-of-the-art methods in high-quality portrait style transfer and flexible style control.

> 借鉴了：[toonify](https://deepai.org/machine-learning-model/toonify) StyleGAN Pixel2style2pixel(psp) 


[github repo](https://github.com/williamyang1991/VToonify)

[huggingface demo](https://huggingface.co/spaces/PKUWilliamYang/VToonify)

[VToonify colab demo](https://colab.research.google.com/github/williamyang1991/VToonify/blob/master/notebooks/inference_playground.ipynb)

[DualStyleGAN colab demo](https://colab.research.google.com/github/williamyang1991/DualStyleGAN/blob/master/notebooks/inference_playground.ipynb)

[介绍文章](https://www.qbitai.com/2022/10/38490.html)

不重要的：
- https://github.com/williamyang1991
- [StyleGANEX（作者最新repo）](https://github.com/williamyang1991/StyleGANEX)



[Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer](https://arxiv.org/abs/2203.13248) 2022.03

VToonify，继承了DualStyleGAN的优点，并且通过修改DualStyleGAN的风格控制模块将这些特性进一步扩展到视频。


## 部署

[huggingface model files](https://huggingface.co/PKUWilliamYang/VToonify/tree/main)

[models folder structure](https://github.com/williamyang1991/VToonify/tree/main/checkpoint)

> 在 pip 中新增：gradio， ipykernel

```bash
git clone https://github.com/williamyang1991/VToonify.git

cd SageMaker/VToonify/
source activate
conda env create -f ./environment/vtoonify_env.yaml
conda activate vtoonify_env

ipython kernel install --user --name vt


# 下载模型用
sudo amazon-linux-extras install epel -y 
sudo yum-config-manager --enable epel
sudo yum install git-lfs -y

git lfs install
git clone https://huggingface.co/PKUWilliamYang/VToonify
mv VToonify/models/* checkpoint/
```


[查看 dualstylegan style_id](https://huggingface.co/spaces/Gradio-Blocks/DualStyleGAN)

- VToonify-D （DualStyleGAN）
- VToonify-T （Toonify）


## 训练


下载训练要用的部分模型文件，已在上文的 git lfs 实现。


下载文件（训练要用）：
```python
def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../checkpoint/. """
    current_directory = os.getcwd()
    save_path = MODEL_DIR
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

MODEL_PATHS = {
    "directions.npy": {"id": "1HbjmOIOfxqTAVScZOI2m7_tPgMPnc0uM", "name": "directions.npy"},
}

for path in MODEL_PATHS.values():
    download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
    !{download_command}
```
sagemaker 有个地方导致 bug，在 ipykernel 中 pip 装到的不是对应的虚拟环境

使用 gdown 下载 folder

```bash
pip install gdown
gdown --help
gdown https://drive.google.com/drive/folders/1xPo8PcbMXzcUyvwe5liJrfbA5yx4OF1j -O ./checkpoint/cartoon --folder
```

参考：https://pypi.org/project/gdown/


>STYLE CONTROL OPTIONS

--fix_degree: if specified, model is trained with a fixed style degree (no degree adjustment)

--fix_style: if specified, model is trained with a fixed style image (no examplar-based style transfer)

--fix_color: if specified, model is trained with color preservation (no color transfer)

--style_id: the index of the style image (find the mapping between index and the style image here).

--style_degree (default: 0.5): the degree of style.


```bash
# 首先根据 dualstylegan 的训练流程跑一遍
# 之后的操作是基于 已有的 dualstylegan 模型
cp -r DualStyleGAN/checkpoint/ VToonify/checkpoint/demo0706


mv DualStyleGAN/checkpoint/0712 VToonify/checkpoint/0712

cd VToonify

# Eight GPUs are not necessary, one can train the model with a single GPU with larger --iter

#（1）pre-training the encoder (4min)
# 需要：（generator-ITER.pt）（refined_exstyle_code.npy）（提供 saved model name: vtoonify_d_cartoon）
# --iter 30000
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 3000 --stylegan_path ./checkpoint/demo0706/generator-001400.pt --exstyle_path ./checkpoint/demo/exstyle_code.npy \
       --batch 1 --name vtoonify_d_demo0706 --pretrain

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 300 --stylegan_path ./checkpoint/demo0710/generator-000300.pt --exstyle_path ./checkpoint/0712/exstyle_code.npy \
       --batch 1 --name vtoonify_d_0712 --pretrain


# 难道是 refine 的锅？
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 300 --stylegan_path ./checkpoint/0712/generator-000300.pt --exstyle_path ./checkpoint/0712/exstyle_code.npy \
       --batch 1 --name vtoonify_d_0712 --pretrain

#（2）给定预训练编码器的情况下，训练 VToonify-D (3min)
# one can train the model with a single GPU with larger --iter.
# --iter 2000 --batch 4
# 3 batch 都会 out of memory, V100-SXM2 16 GB 显存, 4090
# 这里的操作应该再提高 iter

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 3000 --stylegan_path ./checkpoint/demo0706/generator-001400.pt --exstyle_path ./checkpoint/demo0706/refined_exstyle_code.npy \
       --batch 2 --name vtoonify_d_demo0706 --fix_color


python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 300 --stylegan_path ./checkpoint/07121/generator-000300.pt --exstyle_path ./checkpoint/07121/exstyle_code.npy \
       --batch 2 --name vtoonify_d_07121 --fix_color 


cp checkpoint/vtoonify_d_demo0706/pretrain.pt checkpoint/vtoonify_d_demo0707/pretrain.pt

#（2）单独 style_id style_degree 训练
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 9000 --stylegan_path ./checkpoint/demo0706/generator-001400.pt --exstyle_path ./checkpoint/demo0706/refined_exstyle_code.npy \
       --batch 2 --name vtoonify_d_demo0707 --fix_color --fix_degree --style_degree 0.8 --fix_style --style_id 53
```

```bash
# 推理 (2min)
# 6mb 跑出来 61 mb，跑出来很诡异，应该是训练少了。

python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_07121/vtoonify_s_d.pt --style_id 2
```




参考：
- [Training VToonify 文档](https://github.com/williamyang1991/VToonify/tree/main#2-training-vtoonify)
- [Training DualStyleGAN 文档](https://github.com/williamyang1991/DualStyleGAN#3-training-dualstylegan)


## 推理

| 参数           | 默认值          | 说明                                                |
| -------------- | --------------- | --------------------------------------------------- |
| --padding      | 200 200 200 200 | 双眼中间选取距离来切分图像 left, right, top, bottom |
| --ckpt         | vtoonify_s_d.pt |                                                     |
| --video        |                 | 处理视频                                            |
| --batch_size   | 4               | 处理视频时用                                        |
|                |                 |                                                     |
| --style_id     | 26              |                                                     |
| --style_degree | 0.5             | style degree for VToonify-D                         |
| --backbone     | dualstylegan    | ualstylegan/toonify                                 |
| --output_path  | ./output/       |                                                     |


优质的卡通：

toonify_s026_d0.5.pt

[作者的 quick start:](https://github.com/williamyang1991/VToonify/tree/main/output#readme)

```bash
python style_transfer.py --scale_image

python style_transfer.py --scale_image --content ./data/081680.jpg \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt # specialized model has better performance

python style_transfer.py --content ./data/038648.jpg \
       --scale_image --padding 600 600 600 600 --style_id 77 \
       --ckpt ./checkpoint/vtoonify_d_arcane/vtoonify_s_d.pt 

python style_transfer.py --scale_image --content ./data/529.mp4 --video
```

实际使用

```bash
# 默认的，也是最开始给业务方的 demo
# 不带 --scale_image 跑不通。。。
python style_transfer.py --scale_image --content ./data/demo1.mp4 --video

# 没什么区别。。。
python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt

# 有些抽象
python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_pixar/vtoonify_s052_d0.5.pt
```


0704 第三轮
```bash
# specialized model has better performance (author)
# 没发现明显差别
python style_transfer.py --scale_image --content ./0704data/1-4x5.mp4 --video \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt
python style_transfer.py --scale_image --content ./0704data/2-4x5.mp4 --video --padding 150 150 150 150 \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt
python style_transfer.py --scale_image --content ./0704data/3-4x5.mp4 --video --padding 150 150 150 150 \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt
python style_transfer.py --scale_image --content ./0704data/4-4x5.mp4 --video --padding 150 150 150 150 \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt
python style_transfer.py --scale_image --content ./0704data/5-4x5.mp4 --video --padding 300 300 300 300 \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt
```

0704 第四轮 44->

```bash
# 更换 style_id
# --style_id 299 大眼睛显得诡异 （在卡通转卡通上）
# --style_id 8 牙齿很有问题
python style_transfer.py --scale_image --content ./0704data/1-4x5.mp4 --video --style_id 8
python style_transfer.py --scale_image --content ./0704data/2-4x5.mp4 --video --padding 150 150 150 150 --style_id 299
python style_transfer.py --scale_image --content ./0704data/3-4x5.mp4 --video --padding 150 150 150 150 --style_id 299
python style_transfer.py --scale_image --content ./0704data/4-4x5.mp4 --video --padding 150 150 150 150 --style_id 299
python style_transfer.py --scale_image --content ./0704data/5-4x5.mp4 --video --style_id 8 --padding 300 300 300 300
```

0707

```bash
python style_transfer.py --scale_image --content ./data/demo1.mp4 --video --style_id 53\
       --ckpt ./checkpoint/vtoonify_d_demo0706/vtoonify_s_d.pt

python style_transfer.py --scale_image --content ./0704data/1-4x5.mp4 --video --style_id 53\
       --ckpt ./checkpoint/vtoonify_d_demo0706/vtoonify_s_d.pt

python style_transfer.py --scale_image --content ./0704data/1-4x5.mp4 --video --style_id 53\
       --ckpt ./checkpoint/vtoonify_d_demo0707/vtoonify.pt
```




style_transfer 与

colab 

```python
from vtoonify_model import Model
model = Model('cuda')

vtoonify_button.click(fn=model.video_tooniy,
                    inputs=[aligned_video, instyle, exstyle, style_degree, style_type],
                    outputs=[result_video, output_info])
```

## 论文笔记

VToonify combines the merits of the StyleGAN-based framework and the image translation framework.

We first analyze the translation equivariance in StyleGAN which constitutes our key solution to overcome the fixed- crop limitation.


利用相机运动保持转换前后的一致性，抑制闪烁

前后帧之间通过光流抑制闪烁

限制：无法处理极端角度、脸部遮挡、重影、背景闪烁



## 工作流

数据集获取：
[iCartoonFace](https://paperswithcode.com/dataset/icartoonface)
[google cartoonset](https://google.github.io/cartoonset/download.html)

`vtoonify_model Model` 有着一个对功能的封装 gradio 中使用



## 一些说明

FFHQ 是一个高清人脸数据集，并使用了dlib进行人脸对齐和裁剪

> batch_size: 4 占用显存 13360，高质量视频占用的还要更多。

> 默认的 ckpt 和 style_id 26 是女性，虽然男性也适用，但不合适

> backbone 选用 dualstylegan 更好，四周差异比较不明显

```bash
Load options
backbone: dualstylegan
batch_size: 4
ckpt: ./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt
color_transfer: False
content: ./0704data/1-4x5.mp4
cpu: False
exstyle_path: ./checkpoint/vtoonify_d_cartoon/exstyle_code.npy
faceparsing_path: ./checkpoint/faceparsing.pth
output_path: ./output/
padding: [200, 200, 200, 200]
parsing_map_path: None
scale_image: True
style_degree: 0.5
style_encoder_path: ./checkpoint/encoder.pt
style_id: 26
video: True
```

模型命名 说明
```bash
_sXXX: supports only one fixed style with XXX the index of this style.
_s without XXX means the model supports examplar-based style transfer
_dXXX: supports only a fixed style degree of XXX.
_d without XXX means the model supports style degrees ranging from 0 to 1
_c: supports color transfer.
```




style 说明
```bash
cartoon026:      balanced 
cartoon299:      big eyes 
arcane000:       for female 
arcane077:       for male 
pixar052:                  
caricature039:   big mouth 
caricature068:   balanced  
```



```python
style_ind = style_ind * 0 + args.style_id
# sample pre-saved E_s(s)
style = styles[style_ind]

# 这部分导致报错：RuntimeError: CUDA error: device-side assert triggered

Traceback (most recent call last):
  File "train_vtoonify_d.py", line 516, in <module>
    train(args, generator, discriminator, g_optim, d_optim, g_ema, percept, parsingpredictor, down, pspencoder, directions, styles, device)
  File "train_vtoonify_d.py", line 236, in train
    style = styles[style_ind]
RuntimeError: CUDA error: device-side assert triggered

# 使用 CUDA_LAUNCH_BLOCKING=1 跑，并把代码.cuda 去掉，放到 CPU 上

# CPU 上才会出现详细的报错。。。
```






--------------------------

other

百度做的类似功能，四周变异更大

https://ai.baidu.com/tech/imageprocess/selfie_anime

