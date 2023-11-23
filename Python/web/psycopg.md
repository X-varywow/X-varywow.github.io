
PostgreSQL database adapter for Python

1.3k star https://github.com/psycopg/psycopg


psql citus 相关请参考： [应用/大数据/psql](/应用/大数据/psql)

</br>

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
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                except:
                    error = traceback.format_exc()
                    logger.error(f"PostgresConnection _warm_up citus failed by: {error}")
            time.sleep(1)

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

</br>

<u>对于这种 IO 密集型任务，使用多线程而不是多进程，有着更小的 IO 开销（线程之间共享内存空间，成本更低的上下文切换）</u>

测试了一下，性能提升了10倍，对于一批1000次点查的 SQL


> daemon 表示一个进程是否是守护进程；主线程中创建的线程默认为 daemon = False </br>
> 3.10 使用 con_thread.daemon = True 而不是 con_thread.setDaemon(True)




</br>

方式二：使用 DBUtils，[官方文档](https://webwareforpython.github.io/DBUtils/main.html)

```python

```








---------

参考资料：
- [psycopg官方文档](https://www.psycopg.org/psycopg3/docs/basic/usage.html)
- https://www.psycopg.org/psycopg3/docs/advanced/pool.html