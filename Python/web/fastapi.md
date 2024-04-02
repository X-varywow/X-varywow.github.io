

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


引入 pydantic 是为了防止传入不规范的数据



</br>

## _lifespan_

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI


def fake_answer_to_everything_ml_model(x: float):
    return x * 42


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def predict(x: float):
    result = ml_models["answer_to_everything"](x)
    return {"result": result}
```

deprecated:

```python
@app.on_event("startup")
async def startup_event():
    items["foo"] = {"name": "Fighters"}
    items["bar"] = {"name": "Tenders"}


@app.on_event("shutdown")
def shutdown_event():
    with open("log.txt", mode="a") as log:
        log.write("Application shutdown")
```

demo, 使用监控：

```bash
pip install prometheus-fastapi-instrymentator


mkdir "monitor"
export prometheus_multiproc_dir="monitor"
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event('startup')
def start_prometheus():
    Instrumentator().instrument(app).expose(
        app,
        endpoint=f'{app_name}/metrics',
        tags=['system']
    )


# 引入 grafana 的接口： /{app_name}/metrics
# curl http://0.0.0.0:8888/{app_name}/metrics
```


</br>

## _response_


返回 html

```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        request=request, name="item.html", context={"id": id}
    )
```

直接渲染

```python
from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html', 'xml'])
)

template = env.get_template('index.html')
content = template.render(context={})  # 你可以在这里传递需要的上下文

return HTMLResponse(content)
```



</br>

## _other_


fastapi 与 numpy 一起使用时，

报错如下：`ValueError: [TypeError("'numpy.int64' object is not iterable"),`

堆栈全是 fastapi encoder 的错误信息，最后发现：返回的元素（list中包含）了 numpy.int64 类型。




-------------------------------

参考资料：
- https://fastapi.tiangolo.com/zh/
- https://juejin.cn/post/6844904051327369224