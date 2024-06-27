
PostgreSQL database adapter for Python

1.3k star https://github.com/psycopg/psycopg


psql citus 相关请参考： [应用/大数据/psql](/应用/大数据/psql)


```python
pip install "psycopg[binary,pool]"
```

几个重要方法：
- psycopg.connect(**config) as conn
- conn.cursor() as cur
- cur.execute(sql)
- cur.fetchone()
- cur.fetchmany(size), size 表示要取多少行, 每行会包装成元组
- cur.fetchall()
- cur.rowcount 返回变动行数，常用于 cur.execute("update")，每次 execute 会刷新计数




```sql
SELECT schema_name FROM information_schema.schemata

SELECT table_name FROM information_schema.tables WHERE table_schema='public'
```




------------


## _cursor_


```python
import psycopg

# Connect to an existing database
with psycopg.connect("dbname=test user=postgres") as conn:

    # Open a cursor to perform database operations
    with conn.cursor() as cur:

        # Execute a command: this creates a new table
        cur.execute("""
            CREATE TABLE test (
                id serial PRIMARY KEY,
                num integer,
                data text)
            """)

        # Pass data to fill a query placeholders and let Psycopg perform
        # the correct conversion (no SQL injections!)
        cur.execute(
            "INSERT INTO test (num, data) VALUES (%s, %s)",
            (100, "abc'def"))

        # Query the database and obtain data as Python objects.
        cur.execute("SELECT * FROM test")
        cur.fetchone()
        # will return (1, 100, "abc'def")

        # You can use `cur.fetchmany()`, `cur.fetchall()` to return a list
        # of several records, or even iterate on the cursor
        for record in cur:
            print(record)

        # Make the changes to the database persistent
        conn.commit()
```


</br>

## _ConnectionPool_

使用连接池，避免反复连接（频繁的连接和断开会导致大量的资源浪费）


```python
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg_pool import ConnectionPool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logging.getLogger("psycopg.pool").setLevel(logging.INFO)

# 默认连接池的最小连接数为 4
pool = ConnectionPool(min_size=2) 

pool.wait()
logging.info("pool ready")

def square(n):
    with pool.connection() as conn:
        time.sleep(1)
        rec = conn.execute("SELECT %s * %s", (n, n)).fetchone()
        logging.info(f"The square of {n} is {rec[0]}.")

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(square, n) for n in range(4)]
    for future in as_completed(futures):
        future.result()
```

- conn = pool.get_conn() 连接池中获取连接
- pool.get_stats() 返回连接池状态
  - pool_size, 连接池当前连接数
  - pool_available， 连接池当前可用连接数
  - requests_waiting, 等待连接的请求数量



ConnectionPool 基于同步模型，

AsyncConnectionPool 是基于异步实现的连接池，在 async with 的异步上下文中使用，能支持大量的并发请求。




</br>

## _COPY_


写数据：

```python
records = [(10, 20, "hello"), (40, None, "world")]

with cursor.copy("COPY out_table_name (col1, col2, col3) FROM STDIN") as copy:
    for record in records:
        copy.write_row(record)
```

读数据：

```python
with cur.copy("COPY (VALUES (10::int, current_date)) TO STDOUT") as copy:
    copy.set_types(["int4", "date"])
    for row in copy.rows():
        print(row)  # (10, datetime.date(2046, 12, 24))
```

参考：https://www.psycopg.org/psycopg3/docs/basic/copy.html



</br>

## _transactions_

默认情况下，不会自动提交事务，即 conn.autocommit = False;

设为 autocommit = True 后每次执行 SQL 语句会自动提交。

执行多个操作且需要保证事务，可以使用如下：

```python 
with pg_pool.connection() as conn:
    conn.autocommit = False
    try:
        for i in range(n):
            conn.execute(sql)
    except:
        conn.rollback()
        raise
    else:
        conn.commit()
```



</br>

## _多线程&连接池_

方式一：

[concurrent.futures 官方文档](https://docs.python.org/zh-cn/3/library/concurrent.futures.html)

```python
import traceback
from psycopg_pool import ConnectionPool
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from loguru import logger
import time

class my_pool:
    def __init__(self, ...):
        self.pool = ConnectionPool(...)
        con_thread = threading.Thread(target=self._warm_up)

        # 3.10 已弃用
        con_thread.setDaemon(True)
        con_thread.start()

    def _warm_up(self):
        while True:
            with self.pool.connection() as conn:
                conn.autocommit = True
                try:
                    # with conn.cursor() as cursor:
                    #     cursor.execute("SELECT 1")
                    self.pool.check()
                except:
                    error = traceback.format_exc()
                    logger.error(f"PostgresConnection _warm_up citus failed by: {error}")
            time.sleep(20)

    def query(self, sql, row_to_dict=False):
        res = []
        with self.pool.connection() as conn:
            conn.autocommit = True
            try:
                with conn.cursor(row_factory=dict_row) if row_to_dict else conn.cursor() as cursor:
                    cursor.execute(sql)
                    res = cursor.fetchall()
            except:
                error = traceback.format_exc()
                logger.error(f"PostgresConnection query citus with sql: {sql} failed by: {error}")
        return res

EXECUTOR = ThreadPoolExecutor(max_workers=32)

pool = my_pool(**config)

def parallel_query(sqls):
    futures = [EXECUTOR.submit(pool.query, sql) for sql in sqls]
    done, _ = wait(futures, return_when=ALL_COMPLETED, timeout=3)
    res = []
    for future in done:
        res = future.result()
        if res:
            res.append(result)
    return res
```

psycopg 源码：

```python
def check_connection(conn: CT) -> None:
    """
    A simple check to verify that a connection is still working.

    Return quietly if the connection is still working, otherwise raise
    an exception.

    Used internally by `check()`, but also available for client usage,
    for instance as `!check` callback when a pool is created.
    """
    if conn.autocommit:
        conn.execute("SELECT 1")
    else:
        conn.autocommit = True
        try:
            conn.execute("SELECT 1")
        finally:
            conn.autocommit = False
```







</br>

<u>对于这种 IO 密集型任务，使用多线程而不是多进程，有着更小的 IO 开销（线程之间共享内存空间，成本更低的上下文切换）</u>

测试了一下，性能提升了10倍，对于一批1000次点查的 SQL

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231125165120.png">


?> daemon 表示一个进程是否是守护进程；主线程中创建的线程默认为 daemon = False </br>
守护线程，在主程序退出时不会等待守护线程的完全，相当于一个优先级低的线程，常用于：程序自检、清理。


3.10 使用 con_thread.daemon = True 而不是 con_thread.setDaemon(True)

```python
# 简洁写法
threading.Thread(target=worker, daemon=True).start()
```



</br>

方式二：使用 DBUtils，[官方文档](https://webwareforpython.github.io/DBUtils/main.html)

```python

```


</br>

## _other_


查看表结构信息：

```sql
SELECT
    column_name, 
    data_type
FROM INFORMATION_SCHEMA.COLUMNS
WHERE 
    "table_schema" = 'scheme_name' 
    and "table_name" = 'table_name';
```

----------

避免 sql 注入:


如果直接使用拼接的 SQL，它可能会导致 SQL 注入攻击：

```python
user_input = "1; DROP TABLE users;"

sql = f"SELECT * FROM my_table WHERE my_column = {user_input}"
```

使用 %s 进行参数化查询 (被数据库视为普通的字符串值，而不是一部分可执行的SQL代码)

```python
# 安全的做法
cursor.execute("SELECT * FROM my_table WHERE my_column = %s", (user_input,))
```


----------

关于使用 sqlachemy 还是 psycopg

sqlachemy 提供了一个强大的对象关系映射（ORM），用于简化数据库操作，减少编写原生 SQL 语句。

psycopg 可直接编写 SQL 进行数据库操作，最大化性能，是 python 中最常用的 postgresql 适配器。

---------

psycopg 默认的超时时间 30s

The default maximum time in seconds that a client can wait to receive a connection from the pool 

-> 超时的原因不仅可能是连接问题，还可能是 sql 执行超时的问题

---------

参考资料：
- [psycopg官方文档](https://www.psycopg.org/psycopg3/docs/basic/usage.html)
- https://www.psycopg.org/psycopg3/docs/advanced/pool.html