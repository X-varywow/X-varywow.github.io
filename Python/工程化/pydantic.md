
## _preface_

pydantic 是一个快速的数据验证和传递的库。

```python
from pydantic import BaseModel

class ApiRequest(BaseModel):
    user_id:int = 21172986
    app_id:int = 1001
    
d = {'user_id':  123}

req = ApiRequest(**d)
  
req.model_dump()
```


```python
from datetime import datetime
from pydantic import BaseModel, PositiveInt
from typing import List, Optional,Tuple


class Delivery(BaseModel):
    timestamp: datetime
    dimensions: Tuple[int, int]
    arr: Optional[list] = field(default_factory=list)


m = Delivery(timestamp='2020-01-02T03:04:05Z', dimensions=['10', '20'])
print(repr(m.timestamp))
#> datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=TzInfo(UTC))
print(m.dimensions)
#> (10, 20)
```


</br>

## _数据类型_

- int
- float
- str
- bool
- datetime
- Tuple[int, int]
- dict[str, PositiveInt]
- Optional[str] (支持传递 None)（默认就支持不传全部参数, 字面意思不对）
- datetime | None
- List[int] (或者 `list = field(default_factory=list)`)
- 支持递归 BaseModel



--------

```python
class Item(BaseModel):
    id: Optional[int] = None
    id2: int = 1

# Optional 可选参数
# id2 写法要么不传过来，要么传过来不能为 None
```

传过来 null 会报错：422 Unprocessable Entity




--------

动态属性，新增属性而不必是定义中的属性

```python
from pydantic import BaseModel

class DataModel(BaseModel):
    feature: str

data = DataModel(feature='some value')
data.feature = 'new value'  # 设置已存在的属性值
data.new_feature = 'another value'  # 动态添加新属性
```





</br>

## _类型验证_

基于 rust 实现，速度较快

```python
# continuing the above example...

from pydantic import ValidationError


class User(BaseModel):
    id: int
    name: str = 'John Doe'
    signup_ts: datetime | None
    tastes: dict[str, PositiveInt]


external_data = {'id': 'not an int', 'tastes': {}}  

try:
    User(**external_data)  
except ValidationError as e:
    print(e.errors())
    """
    [
        {
            'type': 'int_parsing',
            'loc': ('id',),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': 'not an int',
            'url': 'https://errors.pydantic.dev/2/v/int_parsing',
        },
        {
            'type': 'missing',
            'loc': ('signup_ts',),
            'msg': 'Field required',
            'input': {'id': 'not an int', 'tastes': {}},
            'url': 'https://errors.pydantic.dev/2/v/missing',
        },
    ]
    """

```

使用 field 数据验证

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., alias='username', default='Guest', title='User Name')
    age: int = Field(..., gt=0, lt=100)
```

- Field(..., alias='username', default='Guest', title='User Name') 表示 name 字段有一个别名 username，如果没有提供 name 的值，则默认为 'Guest'。title 参数提供了一个更友好的字段名称，用于生成文档或错误消息。
- Field(..., gt=0, lt=100) 表示 age 字段必须大于 0 且小于 100，这里使用了 gt（greater than）和 lt（less than）参数来指定验证规则。




</br>

## _序列化_

```python
from datetime import datetime

from pydantic import BaseModel


class Meeting(BaseModel):
    when: datetime
    where: bytes
    why: str = 'No idea'


m = Meeting(when='2020-01-01T12:00', where='home')
print(m.model_dump(exclude_unset=True))
#> {'when': datetime.datetime(2020, 1, 1, 12, 0), 'where': b'home'}
print(m.model_dump(exclude={'where'}, mode='json'))
#> {'when': '2020-01-01T12:00:00', 'why': 'No idea'}
print(m.model_dump_json(exclude_defaults=True))
#> {"when":"2020-01-01T12:00:00","where":"home"}

```

--------------

参考资料：
- https://pydantic.dev/
- https://docs.pydantic.dev/latest/