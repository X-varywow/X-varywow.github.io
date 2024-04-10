

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


## other

发生如下报错：

OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.


```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

---------------------

conda 发生如下报错：

Platform: win-64
Collecting package metadata (repodata.json): | Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x0000023A1EB35890>: Failed to resolve 'conda.anaconda.org' ([Errno 11002] getaddrinfo failed)")': /nvidia/win-64/repodata.json.zst

\ Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x0000023A1EB07B10>: Failed to resolve 'conda.anaconda.org' ([Errno 11002] getaddrinfo failed)")': /nvidia/win-64/repodata.json.zst


