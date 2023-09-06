


## _ONNX Runtime_


ONNX Runtime是一个 **高性能的推理引擎**，用于在各种硬件平台上执行深度学习模型的推理。由微软开发并开源的，旨在提供一个统一的推理框架，使开发者能够轻松地在不同的平台上部署和执行深度学习模型。

ONNX Runtime支持ONNX（Open Neural Network Exchange）格式，这是一种开放的深度学习模型交换格式。ONNX是由多家公司共同创建的，旨在解决深度学习模型在不同框架之间的互操作性问题。ONNX Runtime可以加载和执行ONNX格式的模型，使得开发者可以使用不同的深度学习框架进行训练，并在不同的硬件上进行推理。

优点：
- 高性能和低延迟
  - 能够充分利用硬件加速器（如GPU、TPU等）的计算能力。
- 跨平台
  - 可以在多种操作系统（如Windows、Linux等）和硬件平台上运行。此外，ONNX Runtime还支持模型优化和量化，以提高推理速度和减少模型的存储空间。
- 还提供了一些辅助工具和库，用于模型的转换、优化和部署。它可以与各种编程语言（如Python、C++等）和深度学习框架（如PyTorch、TensorFlow等）进行集成，方便开发者使用和扩展。



----------

参考资料：
- https://onnxruntime.ai/
- chatgpt


</br>

## _服务方式_

服务器常驻

[部署期间的机器学习推理](https://learn.microsoft.com/zh-cn/azure/cloud-adoption-framework/innovate/best-practices/ml-deployment-inference)


</br>

## _前端界面_

https://streamlit.io/gallery

gradio