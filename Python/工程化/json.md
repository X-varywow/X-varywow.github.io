

## _json_


json.dump - 将Python对象按照JSON格式序列化到文件中

json.dumps - 将Python对象处理成JSON格式的字符串

json.load - 将文件中的JSON数据反序列化成对象

json.loads - 将字符串的内容反序列化成Python对象


示例：从 json 文件中加载数据

```python
import json
from pprint import pprint

file = "./mtn_03.motion3.json"
with open(file) as f:
    res =json.load(f) 

pprint(res)
```

示例：保存信息到 json 文件

```python
res = {}
with open("data.json", "w") as f:
    json.dump(res, f)

# 保存中文文件时，使用如下方式：（不会被编码为 \u6a31 类似的）
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=4)
```






## _other_

[pickle --- Python 对象序列化](https://docs.python.org/zh-cn/3.10/library/pickle.html)

[shelve --- Python 对象持久化](https://docs.python.org/zh-cn/3.10/library/shelve.html)

[sqlite3 --- SQLite 数据库 DB-API 2.0 接口模块](https://docs.python.org/zh-cn/3.10/library/sqlite3.html)


---------------

参考资料：
- [官方文档](https://docs.python.org/zh-cn/3/library/json.html)