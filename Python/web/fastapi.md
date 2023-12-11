

## _preface_

fastapi 是一个用于构建 API 的现代、快速（高性能）的 web 框架，使用 Python 3.6+ 并基于标准的 Python 类型提示。

支持 异步，性能上接近 Go。


- Starlette 负责 web 部分
- Pydantic 负责数据部分


```python
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

```


引入 BaseModel 是为了防止传入不规范的数据



-------------------------------

参考资料：
- https://fastapi.tiangolo.com/zh/
- https://juejin.cn/post/6844904051327369224