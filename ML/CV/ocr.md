

https://github.com/JaidedAI/EasyOCR  21K stars

## 环境准备

windows 11

python 3.11

torch 2.2.2

cuda 12.1

```bash
pip config list

conda config --show-sources

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

## 运行

```python
import easyocr
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
result = reader.readtext('640_new.png')
```

之后会自动下载模型：

Downloading detection model, please wait. This may take several minutes depending upon your network connection.


但还是先从 [modelhub](https://www.jaided.ai/easyocr/modelhub/) 下载到： ~/.EasyOCR/model



> 结果中文识别很捞啊，一张 png 识别很差。。后面看看

