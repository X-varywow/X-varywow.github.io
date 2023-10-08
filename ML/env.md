
## 服务商

_Colab_

- 挂载只有12个小时
- 模型在训练的过程中 有可能会出现连接中断需要重新连接的情况
- 谷歌应用市场里就有一个拓展工具Auto reconnect colab 

语法

```python
#@title STEP 1 安装运行环境


#@markdown CJE为中日英三语模型，CJ为中日双语模型

ADD_AUXILIARY = True #@param {type:"boolean"}


!nvidia-smi


pwd


%run download_model.py
```

[nvidia-smi 更多的介绍](https://blog.csdn.net/C_chuxin/article/details/82993350)

```python
# 查看GPU信息
!/opt/bin/nvidia-smi   # Tesla K80

nvidia-smi

# 每 1 s 显示一次显存
watch -n 1 nvidia-smi

# 提供由Linux内核管理的所有当前运行任务的动态实时统计汇总。它监视 Linux 系统上进程、CPU 和内存的完整利用率
top

# 查看cpu配置
!cat /proc/cpuinfo | grep model\ name

# 查看内存容量
!cat /proc/meminfo | grep MemTotal

# 查看pytorch版本
import torch
print(torch.__version__)

# 查看虚拟机硬盘容量
!df -lh
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

</br>

- 优点：
  - colab 挺好用的，代码写好一点，什么 git clone,run,都写进去，运行也快，除代码啥都不用管了。
- 缺点：
  - 每天GPU运行时间有限制，用多了就连不上去，等第二天。 
  - 100 compute units 也不够用多久，T4 GPU consumes 1.96 compute units per hour
  - 运行依赖，磁盘资源 reconnect 后会重置成默认的

参考资料：
- [Colab 实用教程](https://www.cnblogs.com/zgqcn/p/11186406.html)
- [Google Colab Tutorial 2023](https://colab.research.google.com/drive/1Qi4-BRqZ3qI3x_Jtr5ci_oRvHDMQpdiW?usp=sharing)
- [Colab使用教程（超级详细版）及Colab Pro/Pro+评测](https://www.cnblogs.com/softcorns/p/16369045.html)
- [知乎 - 如何评价 Google Colab 提供的免费 GPU？](https://www.zhihu.com/question/266242493)




</br>

_SageMaker_

[Amazon SageMaker 定价](https://aws.amazon.com/cn/sagemaker/pricing/)

在 Sagemaker 中如何使用 s3：[通过 AWS CLI 使用高级别 (s3) 命令](https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/cli-services-s3-commands.html)

一项完全托管的机器学习服务


ml.p3.2xlarge，3.825 USD/h

[使用生命周期配置脚本自定义笔记本实例](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/notebook-lifecycle-config.html)


</br>

_Azure_

microsoft


## 显卡


_显卡天梯图_


| 型号        | 大概性能 |
| ----------- | -------- |
| GTX 1080ti  | 1        |
| RTX 2080ti  | 2        |
| RTX 3080ti  | 3        |
| telsa T4    |          |
| tesla V100  | 4        |
| NVIDIA A100 | 8        |
|             |          |
|             |          |

_显卡参数表：_
<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230329235350.png">

_显卡性能天梯图：_
<img src="https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/GPUS_Ada_raw_performance3.png?ssl=1">

_显卡每美元算力天梯图：_
<img src="https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/GPUs_Ada_performance_per_dollar6.png?ssl=1">

_显卡选购推荐：_
<img src="https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/gpu_recommendations.png?ssl=1">

参考资料：
- [【主流Nivida显卡深度学习/强化学习/AI算力汇总】](https://blog.csdn.net/weixin_42483745/article/details/125098630)
- [深度学习GPU最全对比，到底谁才是性价比之王？ | 选购指南](https://zhuanlan.zhihu.com/p/61411536)
- [Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)



## 数据下载

（1）wget 方式

```bash
# ljsspeech 数据集
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -jxvf LJSpeech-1.1.tar.bz2

# vctk 数据集
wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip
sudo yum install p7zip
7za x DS_10283_3443.zip
7za x VCTK-Corpus-0.92.zip
# 对于 zip 文件，还可以使用 unzip，不适用于大文件？


# hifi 数据集
wget https://www.openslr.org/resources/109/hi_fi_tts_v0.tar.gz
tar -zxvf hi_fi_tts_v0.tar.gz


wget https://huggingface.co/datasets/Roh/ryanspeech/resolve/main/data/train.tar.gz
tar -zxvf train.tar.gz
```

（2）gdown 方式

```bash
pip install gdown
gdown --help

gdown https://drive.google.com/drive/folders/1xPo8PcbMXzcUyvwe5liJrfbA5yx4OF1j -O ./checkpoint/cartoon --folder
gdown 'https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT'
```

如果格式对不上，采用这种方式：

```python
import os

MODEL_PATHS = {
    "directions.npy": {"id": "1cKZF6ILeokCjsSAGBmummcQh0uRGaC_F", "name": "all_sequences.zip"},
}
MODEL_DIR = "./"

def get_download_model_command(file_id, file_name):
    """
     Get wget download command for downloading the desired model and save to directory ./
    """
    current_directory = os.getcwd()
    save_path = MODEL_DIR
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

for path in MODEL_PATHS.values():
    download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
    !{download_command}
```