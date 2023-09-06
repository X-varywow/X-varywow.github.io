
## preface

- [ ] sd lora 跑通
- [ ] 生成十几张
- [ ] 放到 dualstylegan 中训练
- [ ] 放到 vtoonify 中训练
- [ ] 修改代码，轮次保存部分


## 工作流


### SETP1 数据集准备

https://civitai.com/models/95736/cartoon-portrait


```bash
source activate pytorch_p310

git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui/

sudo amazon-linux-extras install epel -y 
sudo yum-config-manager --enable epel
sudo yum install git-lfs -y

# 下载 SD 底模 + LORA 微调
wget https://civitai.com/api/download/models/94640 -O ./models/Stable-diffusion/majicmixRealistic_v6.safetensors
wget https://civitai.com/api/download/models/102236 -O ./models/Lora/cartoon_portrait_v1.safetensors

# 用的 CUDA 118 不是120
./webui.sh --share

```

真强啊，底模 + LORA；10 张图片，风格单一，试一下






```bash
source activate vtoonify_env


cd ./DualStyleGAN/data
mkdir 0710demo
mkdir 0710demo/images
mkdir 0710demo/images/train
mkdir 0710demo/lmdb
unzip 0710dataset.zip -d ./0710demo/images/train
rm -rf 0710demo/images/train/__MACOSX

# SETP2 DualStyleGAN

```

### SETP2 DualStyleGAN

```bash
# sd + stylegan 环境搭着出问题了，重启一下
# 使用 https://www.gaitubao.com/
# 。。运行完 sd 再跑 vt 跑不了
cd ..
python ./model/stylegan/prepare_data.py --out ./data/demo/lmdb/ --n_worker 4 --size 1024 ./data/demo/images/

python ./model/stylegan/prepare_data.py --out ./data/0712/lmdb/ --n_worker 4 --size 1024 ./data/0712/images/

# FINAL
python ./model/stylegan/prepare_data.py --out ./data/0713/lmdb/ --n_worker 4 --size 1024 ./data/0713/images/








# 生成  finetune.pt 127mb
# 512 到了这里会出问题，所以使用 resample，自动的 resample
# 根据 lmdb finetune stylegan，生成 ./checkpoint/cartoon/finetune-000600.pt
# 结果出不来，又是图片的问题(使用 PIL 解决，)
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 100 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style demo --save_every 100 --augment ./data/demo/lmdb/

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 600 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style 0712 --save_every 100 --augment ./data/0712/lmdb/

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 200 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style 0712 --save_every 100 --augment ./data/0712/lmdb/

#07131， 307 图片，11min
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 600 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style 0713 --save_every 100 --augment ./data/0713/lmdb/

# FINAL
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 1500 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style 0713 --save_every 100 --augment ./data/0713/lmdb/










# 去风格化，生成两个style.npy 11mb
# 这一步可能 images/train 的自动.ipy 导致报错(数字没对上图片数量就是多了个.ipynb_checkpoints文件夹)
python destylize.py --model_name finetune-001200.pt --batch 1 --iter 300 demo
python destylize.py --model_name finetune-000600.pt --batch 1 --iter 300 0712  # 这个
python destylize.py --model_name finetune-000100.pt --batch 1 --iter 100 demo  # 这个
python destylize.py --model_name finetune-000010.pt --batch 1 --iter 10 demo
python destylize.py --model_name finetune-000200.pt --batch 1 --iter 100 demo

#0713， 307 图片，6.5h ,上班跑不了，改动后2h
python destylize.py --model_name finetune-000600.pt --batch 1 --iter 300 0713
# 这步很重要，small --truncation can learn larger structure deformation
python destylize.py --model_name finetune-000600.pt --batch 6 --iter 200 --truncation 0.5 0713

# 重走这个，需要重新后面的 1500 finetune
python destylize.py --model_name finetune-000600.pt --batch 6 --iter 100 --truncation 0.8 0713

# FINAL1（0713晚） 效果不好原因：finetune 多了，但是 truncation 没过来
python destylize.py --model_name finetune-000800.pt --batch 2 --iter 300 --truncation 0.8 0713

# FINAL2
python destylize.py --model_name finetune-001500.pt --batch 2 --iter 300 --truncation 0.8 0713











# 生成  generator.pt 300mb
# 需要3个东西: 两个 style.npy, 自动的 lmdb
#（到这出错）（只根据lmdb 微调generator-pretrain）（已解决，还是 1024 jpg 不会出错）
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 1500 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment demo

# 0713 35min( x2 尝试)
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 1500 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment 0713 --save_begin 100

# 这个应该高，虽然中间会怪异，

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 8000 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --save_begin 100 --augment 0712   # 这个


python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 300 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --save_begin 100 --augment demo   # 这个

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 100 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment demo   # 这个


# FINAL
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 2000 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment 0713 --save_begin 100


# 这里先测一下 2000 的效果哈









# 可选的后续工作
python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/demo/generator-001400.pt demo

python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/0712/generator-000300.pt demo

python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/demo/generator-000300.pt demo   # 这个
python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/demo/generator-000100.pt demo   # 这个

python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/0712/generator-008000.pt 0712

python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/0713/generator-001500.pt 0713


# FINAL
python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/0713/generator-002000.pt 0713

python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/0713-copy/generator-001500.pt 0713



python train_sampler.py demo




























# 推理


python style_map.py --model_name generator-001500.pt --style 0712 --weight 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 


python style_map.py --model_name generator-000600.pt --style 0712

python weight_map.py --model_name generator-001500.pt --style 0712

python weight_map.py --model_name generator.pt --style cartoon --style_id 26

python style_map.py --model_name generator.pt --style cartoon


python weight_map.py --model_name generator-001400.pt --style 0712

python weight_map.py --model_name generator-000800.pt --style 0712

python weight_map.py --model_name generator-000400.pt --style 0712

python weight_map.py --model_name generator-001300.pt --style 0712

python weight_map.py --model_name generator-000900.pt --style 0712

python weight_map.py --model_name generator-000100.pt --style 0712
python weight_map.py --model_name generator-000200.pt --style 0712
python weight_map.py --model_name generator-000500.pt --style 0712
python weight_map.py --model_name generator-000700.pt --style 0712


# 0713
python weight_map.py --model_name generator-001200.pt --style 0713

python weight_map.py --model_name generator-001200.pt --style 0713-copy --style_id 106

python weight_map.py --model_name generator-001500.pt --style 0713-copy --style_id 106

python weight_map.py --model_name generator-001500.pt --style 0713-copy --style_id 106

python weight_map.py --model_name generator-001200.pt --style 0713 --style_id 106 --truncation 0.5

python weight_map.py --model_name generator-001200.pt --style 0713 --style_id 106 --truncation 1

#terrible
python weight_map.py --model_name generator-001600.pt --style 0713 --style_id 106
python weight_map.py --model_name generator-002000.pt --style 0713 --style_id 106

python weight_map.py --model_name generator-000800.pt --style 0713 --style_id 106

python weight_map.py --model_name generator-001200.pt --style 0713

python style_map.py --model_name generator-001500.pt --style 0713

python style_map.py --model_name generator-001400.pt --style 0713

python weight_map.py --model_name generator-001500.pt --style 0713 --style_id 106

python weight_map.py --model_name generator-001000.pt --style 0713 --style_id 106

python weight_map.py --model_name generator-000600.pt --style 0713 --style_id 106

python weight_map.py --model_name generator-000200.pt --style 0713 --style_id 106


# 没有 refine 也背锅, 碎片是 exstyle_code，但测出来不是 refine 的锅。。
#小结：训多了不行 dualstylegan

# 完全不行
python weight_map.py --model_name generator-003000.pt --style 0712

# 很碎片，不好，颜色也损失，不好
python weight_map.py --model_name generator-005000.pt --style 0712

# 更加碎片，不好
python weight_map.py --model_name generator-008000.pt --style 0712

python style_map.py --model_name generator-000600.pt --style 0712 --weight 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 

python style_map.py --model_name generator-000300.pt --style 0712 --weight 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1

python style_map.py --model_name generator-001400.pt --style demo --weight 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.6 0.6 0.6 0.6 1 1 1 1 1 1 1

python style_map.py --model_name generator-000100.pt --style demo

python style_map.py --model_name generator-000300.pt --style demo

python style_map.py --model_name generator-000300.pt --style demo --weight 1 1 1 1 1 1 1 0.6 0.6 0.6 0.6 1 1 1 1 1 1 1

# 使用作者模型生成的，高一批次
python style_map.py --model_name generator.pt --style cartoon --weight 1 1 1 1 1 1 1 0.6 0.6 0.6 0.6 0 0 0 0 0 0 0

python style_map.py --model_name generator-000300.pt --style demo --weight 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.6 0.6 0.6 0.6 1 1 1 1 1 1 1


python style_map.py --model_name generator-000500.pt --style demo --weight 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1 1 1 1 1 1 1 1 1 1 1

python style_map.py --model_name generator-000500.pt --style demo --weight 0.3 0.3 0.3 0.3 0.3 0.3 0.3 1 1 1 1 1 1 1 1 1 1 1

python style_map.py --model_name generator-000500.pt --style demo --weight 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1


# 色彩光泽更好
# 减少微调使其更像人，数据集不好的情况下
python style_map.py --model_name generator-0000.pt --style demo --weight 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 1 1 1 1 1 1 1 

python style_map.py --model_name generator-001400.pt --style demo --weight 1 1 1 1 1 1 1 0.6 0.6 0.6 0.6 1 1 1 1 1 1 1 

```







### SETP3 VToonify

```bash

s_w.repeat

cp -r DualStyleGAN/checkpoint/demo VToonify/checkpoint/demo0710

cp -r DualStyleGAN/checkpoint/0713 VToonify/checkpoint/0713

cd VToonify

# Eight GPUs are not necessary, one can train the model with a single GPU with larger --iter

#（1）pre-training the encoder (4min)
# 需要：（generator-ITER.pt）（refined_exstyle_code.npy）（提供 saved model name: vtoonify_d_cartoon）
# --iter 30000
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 3000 --stylegan_path ./checkpoint/demo0710/generator-000100.pt --exstyle_path ./checkpoint/demo0710/refined_exstyle_code.npy \
       --batch 1 --name vtoonify_d_demo0710 --pretrain


python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 3000 --stylegan_path ./checkpoint/0713/generator-001500.pt --exstyle_path ./checkpoint/0713/exstyle_code.npy \
       --batch 1 --name vtoonify_d_0713 --pretrain

cp -r VToonify/checkpoint/0713 DualStyleGAN/checkpoint/0713-copy

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 30000 --stylegan_path ./checkpoint/0713/generator-001500.pt --exstyle_path ./checkpoint/0713/exstyle_code.npy \
       --batch 1 --name vtoonify_d_0713 --pretrain





# FINAL
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 30000 --stylegan_path ./checkpoint/0713/generator-001500.pt --exstyle_path ./checkpoint/0713/refined_exstyle_code.npy \
       --batch 1 --name vtoonify_d_0714 --pretrain


# 回到原本再进行refine 操作


#（2）给定预训练编码器的情况下，训练 VToonify-D (3min)
# one can train the model with a single GPU with larger --iter.
# --iter 2000 --batch 4
# 3 batch 都会 out of memory, V100-SXM2 16 GB 显存, 4090
# 这里的操作应该再提高 iter

cp checkpoint/vtoonify_d_demo0706/pretrain.pt checkpoint/vtoonify_d_demo0710/pretrain.pt

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 2000 --stylegan_path ./checkpoint/demo0710/generator-000100.pt --exstyle_path ./checkpoint/demo0710/refined_exstyle_code.npy \
       --batch 2 --name vtoonify_d_demo0710 --fix_color

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 2000 --stylegan_path ./checkpoint/0713/generator-001500.pt --exstyle_path ./checkpoint/0713/exstyle_code.npy \
       --batch 2 --name vtoonify_d_0713 --fix_color

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 2000 --stylegan_path ./checkpoint/0713/generator-001500.pt --exstyle_path ./checkpoint/0713/exstyle_code.npy \
       --batch 2 --name vtoonify_d_0713 --fix_color --save_begin 1000 --save_every 500



# FINAL
# weight-default: 1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 3000 --stylegan_path ./checkpoint/0713/generator-001500.pt --exstyle_path ./checkpoint/0713/refined_exstyle_code.npy \
       --batch 2 --name vtoonify_d_0714 --fix_color --save_begin 1000 --save_every 500 \
       --fix_style --style_id 106 --fix_degree --style_degree 0.8



--style_id 6


#（2）单独 style_id style_degree 训练
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 train_vtoonify_d.py \
       --iter 3000 --stylegan_path ./checkpoint/demo0710/generator-000500.pt --exstyle_path ./checkpoint/demo0706/refined_exstyle_code.npy \
       --batch 2 --name vtoonify_d_demo0710 --fix_color --fix_degree --style_degree 0.8 --fix_style --style_id 6
```

```bash
# 推理 (2min)
# 6mb 跑出来 61 mb，跑出来很诡异，应该是训练少了。

python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_demo0706/vtoonify_s_d.pt

python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_0713/vtoonify_s_d.pt

python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_0713/vtoonify_s_d.pt --style_id 106

python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_0713/vtoonify_s_d.pt --style_id 2

# FINAL
python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_0714/vtoonify_s106_d0.8.pt

python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_0714/vtoonify_s106_d0.8_02000.pt

python style_transfer.py --scale_image --content ./data/demo1.mp4 --video \
--ckpt ./checkpoint/vtoonify_d_0714/vtoonify_s106_d0.8_01000.pt --style_id 106


```




## 狐狸工程

换姿势：​https://replicate.com/zsxkib/draggan/api
换表情：​https://aws.amazon.com/cn/blogs/china/using-stable-diffusion-to-improve-game-and-animation-art-production/
生成游戏素材：[MsceneMix 卡通景景大模型](https://civitai.com/models/75879/mscenemix)



## sd 说明


> Clip Skip 是什么

CLIP 是一个先进的神经网络，它将您的提示文本转换为数值表示。

神经网络在处理这种数值表示时表现得非常出色，这也是为什么 SD 的开发者选择 CLIP 作为稳定扩散方法生成图像的三个模型之一。

由于 CLIP 是一个神经网络，这意味着它有很多层。您的提示以简单的方式被数字化，然后通过各层传递。

在第一层之后，您可以得到提示的数值表示，将其输入到第二层，将其输出输入到第三层，依此类推，直到到达最后一层，这就是用于稳定扩散的 CLIP 输出。

这就是滑块值为 1 的情况。但您也可以提前停止，并使用倒数第二层的输出，这就是滑块值为 2 的情况。您停止得越早，神经网络对提示的处理就越少。 有些模型在训练时使用了这种调整方法，因此设置这个值可以帮助在这些模型上获得更好的结果。

Clip Skip = 2 （在 webui setting 中设置），可以使神经网络对提示的处理就越少。

[参考资料](https://zhuanlan.zhihu.com/p/630875053)


--------------

一些插件：

[Mov2mov 生成动画，抠图处理](https://github.com/Scholar01/sd-webui-mov2mov)

通过ControNet控制角色

如使用Canny模型控制画面内容（在适当位置留白填空，按理来说 整个游戏 AI 都可以做）

使用Openpose控制人物姿势

使用Reference Only预处理器控制风格

controlnet tile 模型放大


## 一些结论：

整体的关键：
- 优质的数据集，包含完整头发，
- 先把 dual 训好，关键参数：trunction, destrylize-iter
- 

### DualStyleGAN

从作者的数据来看，


数据集背大锅，，，没有完整头发，（怪不得推理出来头发都是异常的）

风格应该相似



- trunction
  - 训练时，调低该参数，使其学习大的形变
  - style_transfer 推理时，使用默认的即可
- generator-iter
- 较小时，比较贴合真实的人脸，估计色块错位等是 style 文件的问题
- 训练 style.npy 的轮次十分重要
在 0.8 的基础上提高 style 的轮次，


### VToonify



- 不贴合问题（表现 真实 + 虚拟 重影，不贴合，）
  - 是第一步训练 encoder 导致的


从 DualStyleGAN 可以推测 VToonify 的大概效果：头发是否正常，嘴唇普遍厚度，整体风格





### 业务反馈


效果反应：
需要大的形变，一般小形变的投放效果不好，真不真，假不假


### 论文笔记

(2022.10) 的论文，

想学构建网络，改网络，现在还停留在一个个积木黑盒。。


不好改层的原因：时间相关，
temporally-coherent, exemplar-based

- high quailty

将真实人脸编码，stylygan latent space

1) The method needs to cope with unaligned faces and different video sizes to
maintain natural motions. Large video size, or wide angle of view,
can capture more information and prevent the face from moving
out of the frame. 
2) Generating high-resolution videos is desired to
match the widely used HD devices nowadays. 
3) To build a practical
user interaction system, flexible style control should be provided
for users to adjust and select their preference.


Apart from the original high-level style code, we train
an encoder to extract multi-scale content features of the input frame
as the additional content condition to the generator, so that the
key visual information of the frame can be better preserved during
style transfer.  能佐证：encoder 导致形变对不上原像的原因

a flicker suppression loss based on the simulation of camera
motion over a single synthetic data to eliminate flickers


firstOrderMotion 丢失3D信息，不行，

STYLEGAN : prone to artifacts. 易出现重影

the last 11 layers of StyleGAN, which mainly render colors and textures for faces


------------------

参考资料：
- https://github.com/AUTOMATIC1111/stable-diffusion-webui
- [sd 简单介绍](https://www.uisdc.com/stable-diffusion-3)
- [sd 文生图教程](https://www.yuque.com/a-chao/sd/wpfsvcqkq0pgmmpg)