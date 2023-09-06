
tensorboard 是一个可视化工具，依赖于日志文件。可以查看：模型的结构、损失函数变化等。



```python
tensorboard_on = True  #@param {type:"boolean"}

if tensorboard_on:
  %load_ext tensorboard
  %tensorboard --logdir logs/44k
```


运行如下命令即可打开服务：

```bash
tensorboard --logdir=logs
```




参考资料：
- [SpeechT5:Unified-Modal Encoder-Decoder Pre-training for Spoken](https://www.bilibili.com/read/cv14105591)
- [Tensorboard 详解](https://zhuanlan.zhihu.com/p/36946874)
- [如何使用 TensorBoard](https://dl.ypw.io/how-to-use-tensorboard/)

