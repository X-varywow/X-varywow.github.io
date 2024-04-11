
[webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 62.8k

[stable-diffusion](https://github.com/CompVis/stable-diffusion) 50.6k

[huggingface sd v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) [huggingface sd v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

[civitai 模型共享](https://civitai.com/) ⭐️

https://civitai.com/models/6424/chilloutmix



## 部署

windows 执行 webui.bat， 会在 venv 目录下新建环境

然后去 huggingface 上下载模型, 

如：
- [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main) 中 safetensors 文件
- [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)

然后将 6.7GB 的模型放到 webui/models/Stable-diffusion 目录下


-----------------

中途发生如下报错：

OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'openai/clip-vit-large-patch14' is the correct path to a directory containing all relevant files for a CLIPTokenizer tokenizer.


之后在 stable-diffusion-webui-master\venv\Lib\site-packages\transformers\models

```bash
git lfs install
git clone https://hf-mirror.com/openai/clip-vit-large-patch14
```

使用的镜像站 https://hf-mirror.com/




（可选）

[xFormers](https://github.com/facebookresearch/xformers) - Toolbox to Accelerate Research on Transformers


```bash
cd E:\stable-diffusion-webui-master\venv\Scripts
.\python.exe -m pip install --upgrade pip
.\pip3.exe install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

安装的 xformers-0.0.25.post1

---------------------


```bash
git clone repo_name

./webui.sh --share
```


## SDXL


（2023.08）

SDXL 1.0 也是所有开放式图像模型中参数量最多的模型之一，它建立在一个创新的新架构上，由一个 35 亿参数的基础模型和一个 66 亿参数的细化模型组成。

论文地址：https://arxiv.org/pdf/2307.01952.pdf

代码地址：https://github.com/Stability-AI/generative-models

官网：https://stablediffusionxl.com/

huggingface 在线：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0



[SDXL1.0评测](https://zhuanlan.zhihu.com/p/646879971)

[深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识](https://zhuanlan.zhihu.com/p/643420260)



## 插件等

</br>

### _controlnet_

> 很有用，但现在自己没场景

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

</br>

### _EbSynth_

[EbSynth 补帧生成动画](https://www.bilibili.com/video/BV1uX4y1H7U3)


</br>

### _ComfyUI_


https://github.com/comfyanonymous/ComfyUI


## OTHER

[LoRA the Explorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer): 使用 SDXL 生成固定风格图片
