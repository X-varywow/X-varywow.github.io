_preface_
--------------

Pillow（[GITHUB链接](https://github.com/python-pillow/Pillow)）是一个对PIL友好的分支，PIL是一个Python图像处理库。

```python
#安装依赖
#在可执行python代码的环境下执行即可
pip install pillow
pip show pillow

#奇怪，但是 pip install PIL 没有结果，应该是取代了PIL
```

一、基础
---------------


**使用Image类:**

```python
# from Pillow import Image 显示 no module
from PIL import Image
im = Image.open("1.png")

print(im.format,im.size,im.mode) # --> PNG (1080, 2340) RGBA

im.show()  # 会打开外部程序展示图片
```

.

| im.mode | 说明                                     |
| ------- | ---------------------------------------- |
| `1`     | 1-bit 像素, 黑和白, 一个像素占用一个byte |
| `L`     | 8-bit 像素, 黑和白                       |
| `P`     | 8-bit 像素, 使用调色板映射到任何其他模式 |
| `RGB`   | 3x8-bit 像素, 真彩色                     |
| `RGBA`  | 4x8-bit 像素, 带透明度掩码的真彩色       |
| `CMYK`  | 4x8-bit 像素, 分色                       |
| `YCbCr` | 3x8-bit 像素, 颜色视频格式               |

.

```python
# 提取子矩阵
box = (100, 100, 400, 400)
region = im.crop(box)

# PixelAccess Class
px = im.load()
print(px[4,4]) #返回（4，4）位置的RGBA值
```

二、实例
------------------------

### 2.1 添加水印

```python
from PIL import Image, ImageDraw, ImageFont

# 读取图片
lena = Image.open("pillow_data/lena.jpeg")

# 使用ImageDraw模块创建水印
draw = ImageDraw.Draw(lena)

# 使用ImageFont创建字体大小
font = ImageFont.truetype(font= "pillow_data/ARIALN.TTF", size= 18)
draw.text((10, 10), 'HELLO WORLD', font=font, fill='white')

# 显示
display(lena)

# 保存
lena.save("./pillow_data/lena_watermark.jpg")
```


### 2.2 字符画

```python
from PIL import Image # PIL 是一个 Python 图像处理库

ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
# 是我们的字符画所使用的字符集，一共有 70 个字符，字符的种类与数量可以自己根据字符画的效果反复调试的

WIDTH = 60 # 字符画的宽
HEIGHT = 45 # 字符画的高


# 将256灰度映射到70个字符上，也就是RGB值转字符的函数：
def get_char(r, g, b, alpha=256):  # alpha透明度
    if alpha == 0:
        return ' '
    length = len(ascii_char)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)  # 计算灰度
    unit = (256.0 + 1) / length
    return ascii_char[int(gray / unit)]  # 不同的灰度对应着不同的字符
    # 通过灰度来区分色块


if __name__ == '__main__':
    img = 'C:\\Users\\16413\\Pictures\\Saved Pictures\\3.jpg' # 图片所在位置
    im = Image.open(img)
    im = im.resize((WIDTH, HEIGHT), Image.NEAREST)
    txt = ""
    for i in range(HEIGHT):
        for j in range(WIDTH):
            txt += get_char(*im.getpixel((j, i))) # 获得相应的字符
        txt += '\n'
    print(txt)  # 打印出字符画
    # 将字符画 写入文件中
    with open("C:\\Users\\16413\\Pictures\\Saved Pictures\\output.txt", 'w') as f:
        f.write(txt)
```

### 2.3 生成验证码

```python
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))

# 随机颜色1:
def rndColor():
    return (random.randint(64, 255), 
            random.randint(64, 255), 
            random.randint(64, 255))

# 随机颜色2:
def rndColor2():
    return (random.randint(32, 127), 
            random.randint(32, 127), 
            random.randint(32, 127))

# 240 x 60:
width = 60 * 4
height = 60

image = Image.new('RGB', (width, height), (255, 255, 255))

# 创建Font对象:
font = ImageFont.truetype(font= "RuiHeiXiTi.otf", size= 36)

# 创建Draw对象:
draw = ImageDraw.Draw(image)

# 填充每个像素:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())

# 输出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())

# 模糊
image = image.filter(ImageFilter.BLUR)

# image.show()
display(image)
```

### 2.4 九宫格化图片

```python
import sys
from PIL import Image, ImageDraw

img = Image.open("../Pillow_exercise/pillow_data/lena.jpeg")
img = img.resize((500, 500))
img_1 = ImageDraw.Draw(img)
w,h = img.size
xline1 = round(h / 3)
yline1 = round(w / 3)
xline2 = round(h / 3) * 2
yline2 = round(w / 3) * 2

img_1.line([xline1, 0, xline1, h], fill="white", width=6)
img_1.line([xline2, 0, xline2, h], fill="white", width=6)
img_1.line([0, yline1, w, yline1], fill="white", width=6)
img_1.line([0, yline2, w, yline2], fill="white", width=6)

display(img)

img.save("lena_new.jpg")
```


参考资料：
- [pillow英文文档](https://pillow.readthedocs.io/en/latest/handbook/index.html)
- [书栈中文文档](https://www.bookstack.cn/books/Pillow-7.0.0-zh)