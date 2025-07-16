

## preface

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




## lifespan

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI


def fake_answer_to_everything_ml_model(x: float):
    return x * 42


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup part
    # Load the ML model
    ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    yield
    
    # shutdown part; eg. close db
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def predict(x: float):
    result = ml_models["answer_to_everything"](x)
    return {"result": result}
```

> [contextlib](https://docs.python.org/zh-cn/3.13/library/contextlib.html), with语句或函数的 上下文工具



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



## response


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



## CORS & API_KEY

跨域限制 `CORS` 是浏览器的一项安全机制, 限定 前端网页只能访问和自己 同源（相同协议 + 域名 + 端口） 的服务器。

这是限制是有浏览器实施的，


去掉所有来源的 cors:

```python
from fastapi.middleware.cors import CORSMiddleware

...

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=False,  # 必须是 False 才能和 "*" 一起使用
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)
```


-----------



```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader


# API Token验证配置
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key"
    )

app.include_router(router=dol_router, prefix=PREFIX, tags=["service"], dependencies=[Depends(get_api_key)])
# 系统路由无需token验证
app.include_router(router=sys_router, prefix=PREFIX, tags=["system"])
```


## other


fastapi 与 numpy 一起使用时，

报错如下：`ValueError: [TypeError("'numpy.int64' object is not iterable"),`

堆栈全是 fastapi encoder 的错误信息，最后发现：api 返回的元素（list中包含）了 numpy.int64 类型。

合理的，当 api 返回信息时，无法将 python 独有的类型送出去，如 type() 的 `<class 'type'>` 类型、numpy 类型

会报错指向 encoders.py 中 jsonable_encoder


-------------------------------

规范：每个路由（即每个 API 接口）都应该有一个唯一的路径和函数

~~不同接口使用同名函数，只有第二个函数会被注册~~（kimi 的错误答案）

实测下来即使处理函数同名，参数也相同，也能正常处理；

原本报错：`500 Internal Server Error`

是内部代码报错的，在外部只表现个 500；服务器的日志信息很重要。


-----------

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

--------------

异步（async def），基于 asyncio 协程机制，使用 事件循环 驱动。适合 IO 密集型任务（如：数据库、HTTP 调用）并发执行。

CPU 密集型任务推荐 同步 或 后台线程。


-------------------------------

参考资料：
- https://fastapi.tiangolo.com/zh/
- https://juejin.cn/post/6844904051327369224
- [fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices)