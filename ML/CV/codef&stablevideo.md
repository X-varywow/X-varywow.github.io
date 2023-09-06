
## _CoDeF_

仓库地址：https://github.com/qiuyu96/CoDeF

展示网页：https://qiuyu96.github.io/CoDeF/

论文地址：https://arxiv.org/abs/2308.07926



感觉不太行，做出来顶多是一个好一点（加上 control-net）的 DCT-Net, 且预置模型不多，数据集难以构造，

原理倒是挺有趣的。

需要找到合适的使用场景。

使用场景：
- 视频风格转换
- 视频中点跟踪，跟踪切分
- 视频超分


使用 sam-track

大小有限制 540 540，fps 有限制15

> 不太行



## _stablevideo_

相关介绍：https://rese1f.github.io/StableVideo/

仓库地址：https://github.com/rese1f/StableVideo

使用 3.11， 牛的

sd 以及各种插件都不熟悉，正好熟悉一

运行成功，带着 webui，效果一般，原理值得借鉴

- sd 生成前景，远景
- 这物品切分，内容填充确实完美


```bash
git clone https://github.com/rese1f/StableVideo.git
cd StableVideo/

conda create -n stablevideo python=3.11 -y
conda activate stablevideo
pip install -r requirements.txt
pip install xformers
pip install basicsr


# sd15 的一些插件？
git lfs install
git clone https://huggingface.co/lllyasviel/ControlNet

mv ControlNet/models/* ./ckpt/

wget https://www.dropbox.com/s/oiyhbiqdws2p6r1/nla_share.zip
unzip

mv nla_share/* ./data
```
