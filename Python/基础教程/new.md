
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


（2）match-case

```python
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _:
            return "Something's wrong with the internet"
```

（3）新的类型联合运算符

```python
def square(number: Union[int, float]) -> Union[int, float]:
    return number ** 2

def square(number: int | float) -> int | float:
    return number ** 2

isinstance(1, int | str)
```


-----------

<u>python 3.11 语法特性：</u>

宣称比 3.10 快 10% ~ 60%，得益于对解释器的多项优化

对字节码解释器进行了改进





-----------

<u>python 3.12 语法特性：</u>

（1）f-string 中支持表达式

（2）每个解释器一个 GIL，充分利用多核； 3.13 将能直接从 python 层调用



----------

`PEP`， Python Enhancement Proposal， Python 改进建议书。

它是一种为 Python 社区提供新功能建议、信息或者设计准则的正式设计文档。

旨在确保所有对 Python 的重大变更、新特性或者政策在实施之前经过充分讨论、记录与审查。


