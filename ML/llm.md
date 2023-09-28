
需求：一个运行在本地（4080 16GB显存）的 llm，实现 chatgpt 的部分功能。


所有方案：

| 模型     | 说明1                                                                         | 备注                                                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPT4All  | 基于 LLaMA 的新型 7B 语言模型                                                 | 50k [访问地址](https://github.com/nomic-ai/gpt4all) [gpt-j-vs-gpt4all](https://sapling.ai/llm/gpt-j-vs-gpt4all) [LocalAI](https://localai.io/basics/getting_started/) |
| llama2   | [github 地址](https://github.com/facebookresearch/llama)                      | 37.8k [Llama2 70B Chatbot 在线体验](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI)                                                                   |
| chatglm2 | https://github.com/THUDM/ChatGLM2-6B                                          | 33k                                                                                                                                                                   |
| Guanaco  | [huggingface地址](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi) | 33B 效果不好，比不上 gpt                                                                                                                                              |
| Qwen     | [huggingface地址](https://huggingface.co/Qwen/Qwen-7B-Chat)                   | 看指标，比前面的都好                                                                                                                                                  |
|          |                                                                               |                                                                                                                                                                       |
| bard     |                                                                               |                                                                                                                                                                       |
| new bing |                                                                               |                                                                                                                                                                       |


</br>

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230820150811.png">

</br>

暂时结论：


20230812 目前比较好的方案是 Qwen，部署测试后效果不理想。还是需要使用 ChatGPT


## ChatGLM

官网：https://chatglm.cn/ 

Github repo: https://github.com/THUDM/ChatGLM-6B

了解 ChatGLM：https://chatglm.cn/blog

------------

预训练架构：
- GLM（自回归填空）
  - 双头注意力，破坏重建式，即对原始文本 mask 再对 mask 部分预测
- GPT（自回归生成）

------------

>从网页上的测试结果来看，完全比不上 chatgpt

```bash
git clone https://github.com/THUDM/ChatGLM2-6B
cd ChatGLM2-6B
```

------------


还有专门的处理代码的大模型：[CodeGeeX2](https://github.com/THUDM/CodeGeeX2)，在 vscode 上装个插件就行，体验也不是很好


## Qwen

(step.1)

安装 cuda toolkit: https://developer.nvidia.com/cuda-downloads

默认流程一套下来（版本 12.2），执行如下指令会没有反应：

```bash
nvcc --version
```

去这里：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin

显示：

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jul_11_03:10:21_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.2, V12.2.128
Build cuda_12.2.r12.2/compiler.33053471_0
```

(step.2)

!> 高版本 12.2 cuda toolkit 对 torch2.01 的 cuda11.8 适用？

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```python
import torch
torch.__version__           #'2.0.1+cu118'
torch.cuda.is_available()   # True
```



(step.插曲)

anaconda 不给 jupyterlab 的快捷方式，启动麻烦。想把它放到 win10 的磁贴中。

```python
import sys
sys.path

# 找到 anaconda 路径： C:\\ProgramData\\anaconda3
```

找到 jupyter-lab 路径：C:\ProgramData\anaconda3\Scripts\jupyter-lab.exe

将快捷方式，创建到开始菜单：C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)

更改个图标，[png转ico](https://png2icojs.com/zh/) [参考1](https://blog.csdn.net/m0_60841773/article/details/127071717)

最终效果：

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230812002622.png">



(step.3)

参考：github- [7B 的模型](https://github.com/QwenLM/Qwen-7B)

整个 pytorch_model.bin 会有 16 GB，模型真大。

使用 BF16 模式，直接把 16GB 显存吃满了。

flash-attention for higher efficiency and lower memory usage. 但是需要相同 CUDA 版本



实际效果：
<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230812225302.png">


个人感觉：
- 还是 GPT-3.5 Turbo 会更好，体现在字数更多、更加详细
- 中心意思上 Qwen 还是不会偏离的
- 使用起来，Qwen-7B 只有 GPT-3.5 Turbo 60% 的效果

</br>

## _后续_

2023-09-28

[Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary)

-------------------

参考资料：
- https://github.com/QwenLM/Qwen-7B
- [通义千问开源大模型Qwen-7B技术报告](https://zhuanlan.zhihu.com/p/648007297)