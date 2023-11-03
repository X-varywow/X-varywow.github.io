
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

官方文档：https://huggingface.co/docs/transformers/index

Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models.