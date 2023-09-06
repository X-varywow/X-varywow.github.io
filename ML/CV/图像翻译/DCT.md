

仓库地址：https://github.com/menyifang/DCT-Net

展示页面：https://menyifang.github.io/projects/DCTNet/DCTNet.html

论文地址：https://arxiv.org/abs/2207.02426

modelscope: [DCT-Net人像卡通化](https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models/summary)

[ModelScope 卡通化模型](https://www.modelscope.cn/topic/8f0cc825a0d34de28de831ea2d348a9a/pub/lowerTasks)

在线体验：https://modelscope.cn/studios/damo/multi-style_portrait_video_stylization/summary


要求使用 tensorflow (>=1.14, training only support tf1.x)


- [ ] 提高分辨率
- [ ] 人像切分

> 这作者也没有对视频的连贯性做约束，只是分帧然后 Tasks.image_portrait_stylization
> 理论上效果不如 VToonify, 实际上在人像运动不剧烈的情况较好，且 VToonify 比较难训


## _1. 部署_

github 示例上 安装的 tf 版本需要 CUDA 10.0 版本，所以 sagemaker 上采用如下安装方式：

```bash
source activate tensorflow2_p310

pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip install gradio


cd SageMaker/DCT-Net
python app.py
```

### 1.1 简单推理

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import gradio as gr

import imageio
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm


model_dict = {
    "日漫风": 'damo/cv_unet_person-image-cartoon_compound-models',
    "3D风": 'damo/cv_unet_person-image-cartoon-3d_compound-models',
    "艺术效果": 'damo/cv_unet_person-image-cartoon-artstyle_compound-models',
    "sd卡通": 'damo/cv_unet_person-image-cartoon-sd-illustration_compound-models',
    "sd漫画": 'damo/cv_unet_person-image-cartoon-sd-design_compound-models',
}



def inference(filepath, style):
    # style = style_dict[style]
    now = time.time()
    
    outpath = f'output_{str(now)[:-8]}.mp4'

    reader = imageio.get_reader(filepath)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(outpath, mode='I', fps=fps, codec='libx264')

    model_name = model_dict[style]
    img_cartoon = pipeline(Tasks.image_portrait_stylization, model=model_name)

    for _, img in tqdm(enumerate(reader)):
        result = img_cartoon(img[..., ::-1])
        res = result[OutputKeys.OUTPUT_IMG]
        writer.append_data(res[..., ::-1].astype(np.uint8))
    writer.close()
    print('finished!')

    return outpath

inference("./assets/video4.mp4", "日漫风")
```

### 1.2 gradio


```python
# app.py
# gradio==3.15.0

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gradio as gr
import imageio
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm
import numpy as np

style_dict = {"日漫风":"anime", "3D风":"3d", "手绘风":"handdrawn", "素描风":"sketch", "艺术效果":"artstyle"}

def inference(filepath, style):
    style = style_dict[style]
    outpath = 'output.mp4'

    reader = imageio.get_reader(filepath)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(outpath, mode='I', fps=fps, codec='libx264')

    if style=="anime":
        style=""
    else:
        style = '-'+style

    model_name = 'damo/cv_unet_person-image-cartoon'+style+'_compound-models'
    img_cartoon = pipeline(Tasks.image_portrait_stylization, model=model_name)

    for _, img in tqdm(enumerate(reader)):
        result = img_cartoon(img[..., ::-1])
        res = result[OutputKeys.OUTPUT_IMG]
        writer.append_data(res[..., ::-1].astype(np.uint8))
    writer.close()
    print('finished!')

    return outpath



css_style = "#fixed_size_img {height: 240px;} "


title = "AI人像视频卡通化"
description = '''
人像、风景、自制视频片段...上传您心仪的短视频，选择对应风格(日漫风，3D风，手绘风等等)，一键即可转换为不同风格的卡通化视频
'''
# examples = [[os.path.dirname(__file__) + './assets/video1.mp4'], [os.path.dirname(__file__) + './assets/video2.mp4'], [os.path.dirname(__file__) + './assets/video3.mov'], [os.path.dirname(__file__) + './assets/video4.mov'],[os.path.dirname(__file__) + './assets/video5.mov'], [os.path.dirname(__file__) + './assets/video6.mov']]
examples = [[os.path.dirname(__file__) + './assets/video1.mp4'], [os.path.dirname(__file__) + './assets/video2.mp4']]


with gr.Blocks(title=title, css=css_style) as demo:
    
    gr.HTML('''
      <div style="text-align: center; max-width: 720px; margin: 0 auto;">
                  <div
                    style="
                      display: inline-flex;
                      align-items: center;
                      gap: 0.8rem;
                      font-size: 1.75rem;
                    "
                  >
                    <h1 style="font-family:  PingFangSC; font-weight: 500; line-height: 1.5em; font-size: 32px; margin-bottom: 7px;">
                          AI人像视频卡通化
                    </h1>
                  </div>
                  
                </div>
      ''')
    gr.Markdown(description)
    with gr.Row():
        radio_style = gr.Radio(label="风格选择", choices=["日漫风", "3D风", "手绘风", "素描风", "艺术效果"], value="日漫风")
    with gr.Row():
        vid_input = gr.inputs.Video(source="upload")
        vid_output = gr.outputs.Video()
    with gr.Row():
        btn_submit = gr.Button(value="一键生成", elem_id="blue_btn")

    examples = gr.Examples(examples=examples, inputs=[vid_input], outputs=vid_output)
    btn_submit.click(inference, inputs=[vid_input, radio_style], outputs=vid_output)


if __name__ == "__main__":
    # demo.launch(share=True)
    # demo.launch()
    demo.launch(enable_queue=True)
```

主要是这两行：（从达摩的 pipeline 中加载模型）

```python
model_name = 'damo/cv_unet_person-image-cartoon'+style+'_compound-models'
img_cartoon = pipeline(Tasks.image_portrait_stylization, model=model_name)
```




## _2. 网络结构_

DCT 层用于提取图像的频域特征，卷积层用于提取空域信息。


摘自：[DCT-Net 水记](https://blog.csdn.net/goryghost/article/details/126863495)




### 2.1 内容矫正网络 CCN

给定源域预训练的 StyleGAN2 和 少量目标域图像，迁移学习训练出一个目标域的 StyleGAN2，用于扩增目标域的数据集。

### 2.2 几何扩展模块 GCM

对源域和扩增出的目标域都进行仿射变换（旋转+平移+放缩+倾斜）

### 2.3 纹理翻译网络 TTN

用未配对图像来训练，U-Net 架构的图像翻译网络



## _3. 训练_

> 需要提供真实人脸和 卡通风格


[训得不好1](https://github.com/LeslieZhoa/DCT-NET.Pytorch/issues/6)，从这个结果来看，需要较大的算力和数据。

>作者回答：论文里是1w张生成+真实的风格图
你的ttn训练的时候的真实图片是经过align和crop的人脸吗？你的ttn做inference的时候如果只是针对人脸部分会好一些吗？如果这些都没有问题的话，尝试着加大batch（分布式训练，增加节点数），增加迭代次数，调整一下损失函数的比重，也许会有不错的结果



## _4. 测试说明_


- 日漫
  - 在脸部没太大运动，并且头发正确的时候，效果很好（对卡通）
- 手绘
  - 在脸部没太大运动，并且头发正确的时候，效果很好（对卡通）
- 3D
  - 效果一般，比不上原视频
- 素描（不试，效果不好）
- 艺术

ModelScope 还有很多风格，先部署一个；

有空再把原本的部署上去


试试自动切图


-------------

参考资料：
- https://www.bilibili.com/video/BV1je411F7s1
- [DCT-Net 水记](https://blog.csdn.net/goryghost/article/details/126863495)