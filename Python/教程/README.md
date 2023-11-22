
欢迎来到 Python 教程


-----------




<u>python 3.10 语法特性：</u>

(1) 更方便的上下文管理

```python
with (
    psycopg.connect(PG_CONFIG) as conn,
    conn.cursor() as cur
):
    cur.execute("select * from test_perf.user_v1 limit 10")
    res = cur.fetchall()
    print(res)
```

终于不用一层套一层地去写上下文的 with 了




------------

一些皆是对象

```python
import model

r = model.rating

# 后续直接使用 r() 执行
```

```python
def hello(name):
    print(f"hello, {name}")

def decorator_func():
    return hello

deco_hello = decorator_func()
deco_hello("Snow")

# 简易的装饰器思想
```

-----------

使用自省等

type()，dir()，getattr()，hasattr()，isinstance()

```python
a = []

# 常用help
help(a)

# 查看源码等信息
a??

# 函数返回对象object的属性和属性值的字典对象
vars()

```