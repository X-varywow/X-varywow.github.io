


openai

> 2023.5，两个月前的，不能再新了

9.9k https://github.com/openai/shap-e

支持：text_to_3d, image_to_3d

这个项目应该是在内部被砍了，然后开源

出来的 3D 模型效果也是不好。


## 部署

```bash
source activate
conda activate pytorch_p310
pip install -e .
```




> STEP1: 准备环境和模型

```python
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自动下载模型：1.78G, 0.89G, 1.26G
xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
```

> STEP2: 推理 3D 模型

```python
batch_size = 4 # 需要30s
guidance_scale = 3.0

# To get the best result, you should remove the background and show only the object of interest to the model.
image = load_image("./shap_e/examples/example_data/corgi.png")

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(images=[image] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)
```

> STEP3:

64 要 9402 显存

```python
render_mode = 'nerf' # you can change this to 'stf' for mesh rendering
size = 64 # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))
```


## 导出

参考教程：https://juejin.cn/post/7238185960059699259

```python
from shap_e.diffusion.sample import sample_latents

# latemts 应该包含着所有的 3D 信息
latents = sample_latents()
```