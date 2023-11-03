


bark 涨的这么快？用 colab 跑了， sagemaker 总是出问题

bark 效果不是很好，展示做得好而已。

部署 bark；效果不是很好，文档不全；算了，


https://huggingface.co/docs/transformers/main/en/model_doc/bark

```python
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
```