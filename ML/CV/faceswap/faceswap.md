


## preface

仓库地址：https://github.com/deepfakes/faceswap

deepfakes 公司开发的，**在两个特定的人身上交换脸**，缺乏泛用性，但是效果好。

逐帧保存为图片, 人脸检测，人脸对齐，人脸分割，训练模型，合并（所以目标脸需要一个视频作为参考训练）

重点在于那个训练模型，就基础的编码-解码结构就行了，

共享一个编码器，分开不同的解码器，就是换脸的大致原理。（然后 AE 结构的各种共享分开等操作，就形成了 faceswap 的变种）

autoencoder 参考：[AutoEncoder](ML/深度学习/autoencoder)



## 环境准备


tensorflow 个大坑货 [tensorflow GPU 支持](https://tensorflow.google.cn/install/gpu?hl=zh-cn#windows_setup)


安装 :
- CUDA 11.2 3GB
- cudnn 680MB   https://developer.nvidia.com/rdp/cudnn-download


很慢，如果用 face swap 自己的安装程序。之后还要修改系统环境变量


```bash
conda create -n faceswap python=3.10 -y

pip install -r ./requirements/requirements_nvidia.txt

conda activate faceswap
cd C:\Users\Administrator\Desktop\faceswap-master
python faceswap.py gui
```





## 使用步骤

准备好了 参考视频、原视频之后，开始训练，合成吧

直接使用 GUI 就行，以下是为了清楚原理。



1、 `python faceswap.py extract`

会从 src 文件夹中提取人脸，自动对齐，并保存到 extract 文件夹


2、`python faceswap.py train`


3、`python faceswap.py convert`



## other

训练需要太久了 12h ~ weeks

tkinter 虽然界面丑了点，还是可以用的。


-------------

参考资料：
- 教程1：https://zhuanlan.zhihu.com/p/376853800
- deepfakes paper: https://arxiv.org/pdf/2001.00179v3.pdf





