
A successor to transformer for LLM

仓库地址：https://github.com/microsoft/unilm/tree/master/retnet



## _说明_

每个RetNet块包含两个模块：多尺度保持（MSR）模块和前馈网络（FFN）模块。

引入位置相关的指数衰减项取代softmax，简化了计算，同时使前步的信息以衰减的形式保留下来。

引入复数空间表达位置信息，取代绝对或相对位置编码，容易转换为递归形式。

另外，保持机制使用多尺度的衰减率，增加了模型的表达能力，并利用GroupNorm的缩放不变性来提高retention层的数值精度。



------------

参考资料：
- https://mp.weixin.qq.com/s/zJsb2vdpEwqXEUgqHmpInA
- https://mp.weixin.qq.com/s/H8HxdkZqY31UPcfwJ8ArVg