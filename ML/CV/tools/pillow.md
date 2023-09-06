## pillow 修改图片大小

参考： [用PIL库修改图片尺寸实例演示，python调整图像大小方法](https://blog.csdn.net/qq_38161040/article/details/87558998)

```bash
pip install pillow
```


```python
p = "./data/"
import os

files = []

for file in os.listdir(p):
    pp = p+file
    if os.path.isfile(pp) and pp.endswith(".png"):
        print(file)
        files.append(pp)
```

```python
from PIL import Image
from tqdm import tqdm

pbar = tqdm(files)

for i, img in enumerate(pbar):
    print(i, img)
    img_switch = Image.open(img) # 读取图片
    img_deal = img_switch.resize((1024,1024),Image.ANTIALIAS) # 转化图片
    img_deal = img_deal.convert('RGB') # 保存为.jpg格式才需要
    img_deal.save( "{:03d}".format(i+1) + "_resize1024.jpg")
```

> 默认颜色的属性是 RGBA，和 RGB 的区别是前者多了透明度的设置。.jpg格式的图片是不支持透明度设置的



## 查看各个通道的值

```python
from PIL import Image as im

i0 = im.open('./assets/img1.png')

i1 = im.open('result.png')

arr0 = list(i0.getdata())

arr1 = list(i1.getdata())

# len(arr)

# len(arr) == 417 * 403

sum([i[3] for i in arr0])

sum([i[3] for i in arr1])
```

得出结论：
- 这张 png 格式为 RGBA
- alpha 通道默认的值全部是 255，即默认是图片不透明
- 经过 BSHM 变化后，alpha 通道数值总和变为原来一般左右，合理。但是其他通道数值也有一些细微的变化。



pillow 其他用法:

```python
img.format

img.size

img.mode

img.getdata()

img.getdata(band = 0)

img_array = np.asarray(img)
img_array[:, :, 0].shape   # R 通道
```