
[webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

[stable-diffusion](https://github.com/CompVis/stable-diffusion) 

[civitai 模型共享](https://civitai.com/)

</br>

## _环境部署_

1. 克隆仓库：[webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 

2. windows 执行 webui.bat， 会在 venv 目录下新建 python 环境

3. huggingface 上下载模型, 

如：
- [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main) 中 safetensors 文件
- [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

然后将 6.7GB 的模型放到 webui/models/Stable-diffusion 目录下

4. 启动 webui.bat


-----------------

中途发生如下报错：

OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'openai/clip-vit-large-patch14' is the correct path to a directory containing all relevant files for a CLIPTokenizer tokenizer.


下载方式1：
```bash
git lfs install
git clone https://hf-mirror.com/openai/clip-vit-large-patch14
```

下载方式2：huggingface 或镜像站 https://hf-mirror.com/  手动下载模型文件

移动到位置：stable-diffusion-webui-master\openai\clip-vit-large-patch14 即可

----------------


（可选：xformers 推理加速）

[xFormers](https://github.com/facebookresearch/xformers) - Toolbox to Accelerate Research on Transformers


```bash
cd E:\stable-diffusion-webui-master\venv\Scripts
.\python.exe -m pip install --upgrade pip
.\pip3.exe install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

安装的 xformers-0.0.25.post1


更改 webui-user.bat ( 对 webui 增设参数) 即可

```bash
set COMMANDLINE_ARGS=--xformers
```

## 常用

去 [civitai](https://civitai.com/) 下载模型

点击图片右下角 i copy 参数进行绘图


setting -> show all pages -> Quicksettings list (setting entries that appear at the top of page rather than in settings tab) 加入 「sd_vae」、「CLIP_stop_at_last_layers」

--------------

- [x] [picx_real](https://civitai.com/models/241415/picxreal)
- [x] [小红书纯欲风格自拍](https://civitai.com/models/68691?modelVersionId=73382)
- [x] [Artist Style | PonyXL | Lora](https://civitai.com/models/351303)


AI 制作普通的游戏素材还是可以的

。。。有点吓人，各种扭曲在一起的时候; 画不好手；其他都挺好的







</br>

## _插件等_








### controlnet

[ControlNet精准控制AI绘画教程](https://zhuanlan.zhihu.com/p/608499305)

[使用 diffusers 训练你自己的 ControlNet](https://huggingface.co/blog/zh/train-your-controlnet)

[Huggingface ControlNet](https://huggingface.co/lllyasviel/ControlNet)

仓库地址：https://github.com/lllyasviel/ControlNet


- 控制方式
  - Canny edge detection
  - Midas depth estimation
  - HED edge detection
  - M-LSD line detection
  - normal map
  - OpenPose pose detection
  - human scribbles
  - semantic segmentation
  - Openpose’s pose detection



还可以把文字藏到图片中

[controlnet-qrcode](https://civitai.com/models/90940/controlnet-qr-pattern-qr-codes)

<img src="https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/646a6cf5-9cd8-443a-ad53-69b7feca02c8/width=450/6.jpeg">


### EbSynth

[EbSynth 补帧生成动画](https://www.bilibili.com/video/BV1uX4y1H7U3)




### ComfyUI


https://github.com/comfyanonymous/ComfyUI

</br>

## _OTHER_

[LoRA the Explorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer): 使用 SDXL 生成固定风格图片


---------------

参考资料：
- [sd civitai 使用](https://blog.csdn.net/sinat_26917383/article/details/131037406)
- [Stable Diffusion WebUI 显示 VAE 模型、CLIP 终止层数](https://zhuanlan.zhihu.com/p/645264967)