
https://github.com/williamyang1991/DualStyleGAN

[colab demo](https://colab.research.google.com/github/williamyang1991/DualStyleGAN/blob/master/notebooks/inference_playground.ipynb)

>这个用的 CUDA10 是因为 stylegan2 用的 CUDA10, 一些新机型会不适配

## 部署

参考：VToonify 中的部署

## 训练


下载文件（训练要用）：
```python
def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../checkpoint/. """
    current_directory = os.getcwd()
    save_path = MODEL_DIR
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

MODEL_PATHS = {
    "stylegan2-ffhq-config-f.pt": {"id": "1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT", "name": "stylegan2-ffhq-config-f.pt"}, 
    "directions.npy": {"id": "1HbjmOIOfxqTAVScZOI2m7_tPgMPnc0uM", "name": "directions.npy"},
    "generator-pretrain.pt": {"id": "1j8sIvQZYW5rZ0v1SDMn2VEJFqfRjMW3f", "name": "generator-pretrain.pt"}, 
}

for path in MODEL_PATHS.values():
    download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
    !{download_command}
```



```bash
rm -rf cartoon/
mkdir cartoon
mkdir cartoon/images
mkdir cartoon/images/train
mkdir cartoon/lmdb
unzip 317.zip -d images/train
```

时间为 317 张图片测算的时间

```bash
# 准备数据，来生成 lmdb 数据（需创建 lmdb 文件夹）(20s)
# 训练素材文件：./data/cartoon/images/
# 用 512 会报错。。。
python ./model/stylegan/prepare_data.py --out ./data/cartoon/lmdb/ --n_worker 4 --size 1024 ./data/cartoon/images/

python ./model/stylegan/prepare_data.py --out ./data/0712/lmdb/ --n_worker 4 --size 1024 ./data/0712/images/

# Fine-tune StyleGAN（根据 lmdb 数据）(12min)
# load model: ./checkpoint/stylegan2-ffhq-config-f.pt
# 修改 自动的 .ipynb_checkpoints 引起报错，加了个 isfile
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 800 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon --augment ./data/cartoon/lmdb/

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 200 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style 0712 --augment ./data/0712/lmdb/

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 200 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon --augment ./data/cartoon/lmdb/

# 之后模型文件： ./checkpoint/cartoon/finetune-000600.pt

# 去风格化 (7 h)
# For styles severely different from real faces, set --truncation to small value like 0.5 to make the results more photo-realistic (it enables DualStyleGAN to learn larger structrue deformations
# vgg16-397923af.pth
# python destylize.py --model_name finetune-000600.pt --batch 1 --iter 300 cartoon
# 这里更改 batch 会导致问题，style_id 会在推导的时候对不上

python destylize.py --model_name finetune-000800.pt --batch 1 --iter 300 cartoon

python destylize.py --model_name finetune-000200.pt --batch 1 --iter 100 0712

python destylize.py --model_name finetune-000200.pt --batch 1 --iter 200 cartoon #0711晚上运行到这
# ./checkpoint/cartoon/instyle_code.npy and ./checkpoint/cartoon/exstyle_code.npy

# 【跳过】，generator-pretrain.pt 已经提供了，是 FFHQ 数据集出来的模型
# Pretrain DualStyleGAN on FFHQ
python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 pretrain_dualstylegan.py --iter 3000 --batch 4 ./data/ffhq/lmdb/

# 与 npy 不耦合
# Fine-Tune DualStyleGAN on Target Domain (40min)
# users may select the most balanced one from 1000-1500. We use 1400 for our paper experiments.

# 还得是jpg，有时 RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
# 生成 generator

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 1500 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment cartoon

python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 finetune_dualstylegan.py --iter 300 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt --style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment 0712

# 生成的模型文件：./checkpoint/cartoon/generator-ITER.pt
```

>可选操作：Latent Optimization and Sampling

```bash
# Refine the color and structure styles to better fit the example style images.
# 生成 refined_exstyle_code.npy
python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/cartoon/generator-001400.pt cartoon

python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/0712/generator-000300.pt 0712

# Training sampling network. 
# Train a sampling network to map unit Gaussian noises to the distribution of extrinsic style codes
# 使用 exstyle_code.npy 生成 ./checkpoint/DATASET_NAME/sampler.pt
python train_sampler.py cartoon

python train_sampler.py 0712
```



## 推理


| 参数           | 默认值                      | 说明                                                         |
| -------------- | --------------------------- | ------------------------------------------------------------ |
| --content      | './data/content/081680.jpg' | 指定输入                                                     |
| --style        | cartoon                     |                                                              |
| --style_id     | 53                          |                                                              |
| --truncation   | 0.75                        | truncation for intrinsic style code (content)                |
| --exstyle_name | None                        | name of the extrinsic style codes                            |
| --name         | cartoon_transfer            | filename to save the generated images                        |
| --truncation   | 0.75                        | truncation for intrinsic style code (content)                |
| --weight       | [0.75]\*7+[1]\*11           | 分别是structur,color 的权重，越重越卡通，应该0.75,0          |
| --wplus        |                             | use original pSp encoder to extract the intrinsic style code |
|                |                             |                                                              |


```bash
python style_transfer.py --model_name generator-001400.pt

python style_map.py --model_name generator-000300.pt --style 0712

python style_transfer.py --model_name generator-001400.pt --style_id 12

# --preserve_color 有效的，尽可能保持头像周围的颜色
python style_transfer.py --model_name generator-001500.pt --preserve_color

# --content './data/content/081680.jpg' 指定输入
# 006750.jpg 0019706.jpg
# 使用 png 会出错。。
python style_transfer.py --model_name generator-001500.pt --content ./data/content/006750.jpg
```


```bash
# 0707, 采用 demo0706 的模型进行推理
# 使用 Dualstylegan 出来的图片都是一样的轮廓，只是配色不同。。。
# 调整 --exstyle_name 改不动轮廓

python style_transfer.py --model_name generator-001500.pt --style_id 53

python style_transfer.py --model_name generator-001500.pt --style_id 23 --exstyle_name exstyle_code.npy

--truncation


--weight 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 1 1 1 1 1 1 1 

[0.75]*7+[1]*11
# 这个结构上会更像卡通

```


## 其他说明

### style_transfer.py 各种图片说明


```python
# (1) 原本的图片
load_image(args.content).to(device)

#（2）pSp重构过，塞进模型的图片（负责 intrinsic style）
img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                   z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
    img_rec = torch.clamp(img_rec.detach(), -1, 1)
    viz += [img_rec]

#（3）参考 style_id 的图片
# exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()
# = ./checkpoint/ cartoon/ exstyle_code.npy

# stylename = list(exstyles.keys())[args.style_id] = Cartoons_00038_07.jpg = ex[53]
# ./data/cartoon/images/train/ Cartoons_00038_07.jpg


if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
    S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
    viz += [S]

#（4）最终生成的图片
img_gen, _ = generator([instyle], exstyle, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                        truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
img_gen = torch.clamp(img_gen.detach(), -1, 1)
viz += [img_gen]
```


### train_sampler 学习

```python
class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument()

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print()
        return self.opt

if __name__ == "__main__":
    device = 'cuda'

    parser = TrainOptions()
    args = parser.parse()
    print('*'*50)


```



### style_id 使用及构建

```python
# 使用：作为 exstyle.npy 的 key 取出对应的 exstyle

exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()
stylename = list(exstyles.keys())[args.style_id]
latent = torch.tensor(exstyles[stylename]).to(device)
exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

img_gen, _ = generator([instyle], exstyle, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                        truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
img_gen = torch.clamp(img_gen.detach(), -1, 1)

# 构建，参考
%python destylize.py --model_name finetune-000600.pt --batch 1 --iter 300 cartoon
# ./checkpoint/cartoon/instyle_code.npy and ./checkpoint/cartoon/exstyle_code.npy
%python refine_exstyle.py --lr_color 0.1 --lr_structure 0.005 --ckpt ./checkpoint/cartoon/generator-001400.pt cartoon

# destylize.py (去风格化)  生成 intrinsic and extrinsic style codes
np.save(os.path.join(args.model_path, args.style, 'exstyle_code.npy'), dict2)

datapath = os.path.join(args.data_path, args.style, 'images/train')
files = os.listdir(datapath) 

for ii in range(0, len(files), args.batch):
    batchfiles = files[ii:ii+args.batch]
    for j in range(imgs.shape[0]):
        dict2[batchfiles[j]] = latent_e[j:j+1].cpu().numpy()

# refine_exstyle.py 没改变

files = list(exstyles_dict.keys())

np.save(os.path.join(args.model_path, args.style, args.model_name), dict) 
dict[batchfiles[j]] = latent[j:j+1].cpu().numpy()
batchfiles = files[ii:ii+args.batch]

# 按照文件名排序来对应 style_id
```

### 生成 style_id 查询图


查看原本 overview.jpg 结构：
```python
print(viz, viz[0].shape)

# 4 张图片，每张图片对应一个4位矩阵
# shape: torch.Size([1, 3, 1024, 1024])
[
    tensor([[[[ ]]]], device = 'cuda:0'),
    tensor([[[[ ]]]], device = 'cuda:0'),
    tensor([[[[ ]]]], device = 'cuda:0'),
    tensor([[[[ ]]]], device = 'cuda:0')
]

```

用到的部分方法：

```python
from torch.nn import functional as F
import torch

# Concatenates the given sequence of seq tensors in the given dimension. 
# All tensors must either have the same shape (except in the concatenating dimension) or be empty.
torch.cat(tensors, dim=0, *, out=None)

# 在原本 dim 所在的位置新增一个维度
torch.unsqueeze(input, dim)
torch.unsqueeze(a, 0)   # a -> [a]


# Applies a 2D adaptive average pooling over an input signal composed of several input planes.
# 对一个输入 Tensor 计算 2D 的自适应平均池化（就是重设图片大小）
# 输入和输出都是 4-D Tensor， 默认是以 NCHW 格式表示的，
#       其中 N 是 batch size, 
#           C 是通道数
#           H 是输入特征的高度
#           W 是输入特征的宽度

# 使用 int 作为 output_size 可能导致图像发生形变
torch.nn.functional.adaptive_avg_pool2d(input, output_size)


torchvision.utils.make_grid(
    tensor: Union[Tensor, List[Tensor]], 
    nrow: int = 8, 
    padding: int = 2, 
    normalize: bool = False, 
    value_range: Optional[Tuple[int, int]] = None, 
    scale_each: bool = False, 
    pad_value: float = 0.0)
```

最终实现：

```python
viz1,viz2 = [], []
for image in images:
    viz1 += [image]
    viz2 += [image]


# 加了一个 unsqueeze [3, 258, 5141] -> [1, 3, 258, 5141]

out1 = torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz1, dim=0), 256), 20).cpu().unsqueeze(0)
out2 = torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz2, dim=0), 256), 20).cpu().unsqueeze(0)

# make_grid [1, 3, 520, 5162] -> [3, 520, 5162]
save_image(torchvision.utils.make_grid(torch.cat([out2, out1], dim=2), 2).cpu(), 
        os.path.join(args.output_path, save_name+'_stylemap.jpg'))
```

### 生成 weigth 对比图



```python
results = []
for i in range(6): # change weights of structure codes 
    for j in range(6): # change weights of color codes
        w = [i/5.0]*7+[j/5.0]*11

        img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True, 
                                truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
        img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 128), -1, 1)
        results += [img_gen]
        
vis = torchvision.utils.make_grid(torch.cat(results, dim=0), 6, 1)
plt.figure(figsize=(10,10),dpi=120)
visualize(vis.cpu())
plt.show()
```



### tqdm 方式


```python
from tqdm import tqdm

pbar = tqdm(range(args.iter))
for i in pbar:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_description(
        (
            f"[{ii:03d}/{len(files):03d}]"
            f" Lperc: {Lperc.item():.3f}; Lnoise: {Lnoise.item():.3f};"
            f" LID: {LID.item():.3f}; Lreg: {Lreg.item():.3f}; lr: {lr:.3f}"
        )
    )
```


### style_fusion

多个风格进行融合

```python
latent = torch.tensor(exstyles[stylename]).repeat(6,1,1).to(device)
latent2 = torch.tensor(exstyles[stylename2]).repeat(6,1,1).to(device)
fuse_weight = torch.arange(6).reshape(6,1,1).to(device) / 5.0
fuse_latent = latent * fuse_weight + latent2 * (1-fuse_weight)
```




### 小结

作者：https://github.com/williamyang1991

- 代码结构
  - TestOptions 搭 parser 的写法
  - 挺好的结构
- 代码理论


