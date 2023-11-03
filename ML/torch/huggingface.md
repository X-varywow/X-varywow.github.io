
这两个经常用到，总结一下

## _datasets_

https://github.com/huggingface/datasets

```bash
pip install datasets
```


```python
from datasets import load_dataset

# Print all the available datasets
from huggingface_hub import list_datasets
print([dataset.id for dataset in list_datasets()][:5])

# Load a dataset and print the first example in the training set
squad_dataset = load_dataset('squad')
print(squad_dataset['train'][0])

# Process the dataset - add a column with the length of the context texts
dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})

# Process the dataset - tokenize the context texts (using a tokenizer from the 🤗 Transformers library)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x['context']), batched=True)
```




</br>

## _transofrmers_

仓库地址：https://github.com/huggingface/transformers

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models.

-----------

transformers.pipeline，常见的深度学习任务都有模型


```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results = classifier(
    ["We are very happy to show you the 🤗 Transformers library.", 
     "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```

-----------

transformers.AutoTokenize，常用于文本处理

主要功能是将输入文本分割成单词或子词，并为每个单词或子词分配一个唯一的标识符。同时，它还提供了一些其他功能，如添加特殊的标记、截断序列、填充序列等。

```python
from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)
```

----------

huggingface optimum 加速推理


----------

参考资料：
- [transformers官方文档](https://huggingface.co/docs/transformers/index)
- chatgpt