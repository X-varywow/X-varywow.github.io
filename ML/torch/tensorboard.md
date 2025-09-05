
tensorboard 是一个 **利用日志构建图像前端的包**, 常用于模型训练过程


```bash
pip install tensorboard 
```

## 使用方法

（1）在 jupyter 中使用：


```python
tensorboard_on = True  #@param {type:"boolean"}

if tensorboard_on:
  %load_ext tensorboard
  %tensorboard --logdir logs/44k

!python train.py
```


（2）在命令行中使用

```bash
tensorboard --logdir=logs
```


## 访问方式

访问 http://localhost:6006/ 或 jupyter 就可以;


----------

sagemaker 中：

如笔记本实例的 url 为：`https://your_note_book_name.notebook.us-east-1.sagemaker.aws/lab/tree/`

将其修改为: `https://your_note_book_name.notebook.us-east-1.sagemaker.aws/proxy/6006/#timeseries`


---------


参考资料：
- [Tensorboard 详解](https://zhuanlan.zhihu.com/p/36946874)
- [如何使用 TensorBoard](https://dl.ypw.io/how-to-use-tensorboard/)

