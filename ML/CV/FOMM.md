
> 给定参考视频，驱动人像图片进行运动，形成动画

https://github.com/AliaksandrSiarohin/first-order-model/

[colab demo](https://colab.research.google.com/github/AliaksandrSiarohin/first-order-model/blob/master/demo.ipynb)，用 ipywidgets 写 ui 的好少。

https://huggingface.co/spaces/abhishek/first-order-motion-model




## 改进


[Motion Representations for Articulated Animation](https://snap-research.github.io/articulated-animation/)

[repo code](https://github.com/snap-research/articulated-animation) [PDF](https://arxiv.org/pdf/2104.11280.pdf) 2021.4


README 上都是 256 256



## MRAA 部署

```bash
python demo.py  --config config/ted384.yaml --driving_video sup-mat/driving.mp4 --source_image img_bk.png --checkpoint checkpoints/ted384.pth
```

需要 https://drive.google.com/drive/folders/1jCeFPqfU_wKNYwof0ONICwsj3xHlr_tb 下载文件


太抽象了，如果参考图片与原始视频不像的时候




## FOMM 部署


```bash
source activate
conda activate pytorch_p310


sudo amazon-linux-extras install epel -y 
sudo yum-config-manager --enable epel
sudo yum install git-lfs -y

git lfs install
git clone https://huggingface.co/spaces/abhishek/first-order-motion-model


cd first-order-motion-model/
pip install -r requirements.txt

pip install gradio

python app.py
```






## 说明：

只适合脸部没有被覆盖的情况（如一些手势、动作会覆盖脸部）；

空间扭曲情况十分常见，只适合比较纯净、两者特征类似的情况。复杂情况很难有效。

但是有效的话（比如作者示例），应用效果还是挺好的。

## 推理


app.py

```python

# ffmpeg  copy, trim and rename
source_image = imageio.imread(image)
reader = imageio.get_reader(video)
fps = reader.get_meta_data()["fps"]

for im in reader:
    frames.append(im)

reader.close()
# 每张图片都变成 256, 256

make_animation

```