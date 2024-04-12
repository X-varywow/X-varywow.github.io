

OpenAI 的 CLIP（对比性语言-图像预训练）是一种多模态模型。它可以理解文本和图像，将已经训练过了的文本和图像匹配起来，学会了识别图像中的内容和描述图像的语言。

[clip](https://github.com/openai/CLIP) 是一个用于 文本、图片特征学习并匹配的模型。sd 中就用到了 clip


无监督训练，需要成对的 <图片，文本> 训练数据


## 原理

通过文本编码器（Transformer 结构）和图像编码器（ResNet 结构和 ViT 结构）对数据编码。

通过最大化 $I_j$ 与 $T_j$ 的余弦相似度来更新网络参数。



## 部署



## （1）提取特征

```python
import torch
import clip
from PIL import Image

# 加载预训练模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图片并进行预处理
image = Image.open("image.jpg")
image_input = preprocess(image).unsqueeze(0).to(device)

# 提取特征向量
with torch.no_grad():
    image_features = model.encode_image(image_input)

# 打印特征向量的形状
print(image_features.shape)
```

torchsize(1, 512)




## （2）label 预测

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```

## (3) clip-vit-large-patch14

https://huggingface.co/openai/clip-vit-large-patch14

```python
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

clip_path = r"E:\stable-diffusion-webui-master\openai\clip-vit-large-patch14"

model = CLIPModel.from_pretrained(clip_path)
processor = CLIPProcessor.from_pretrained(clip_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt")

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

probs
```




------------

参考资料：
- https://zhuanlan.zhihu.com/p/521151393
- https://github.com/openai/CLIP