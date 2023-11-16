位于应用程序的核心逻辑和底层系统之间，用于处理请求、响应和数据流转等中间步骤。

## _cache_

```python
import cachetools

# 条目数为 500， 超时时间为 0.2s
ttl_cache = cachetools.TTLCache(maxsize=500, ttl=0.2)

cache_k = f"{uid}"
v = ttl_cache.get(cache_k)
if v:
    var = v
```

- FIFO cache
- LFU cache
- LRU cache
- MRU cache
- RR cache
- TTL cache
- TLRU cache




</br>

## _pika_



```python
import pika
import psycopg2

# 连接到消息队列
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 创建队列
channel.queue_declare(queue='my_queue')

# 连接到PostgreSQL数据库
conn = psycopg2.connect(database="your_database", user="your_user", password="your_password", host="your_host", port="your_port")
cur = conn.cursor()

# 定义消息处理函数
def callback(ch, method, properties, body):
    # 将消息写入PostgreSQL数据库
    cur.execute("INSERT INTO your_table (column1) VALUES (%s)", (body.decode('utf-8'),))
    conn.commit()

# 监听队列并处理消息
channel.basic_consume(queue='my_queue', on_message_callback=callback, auto_ack=True)

# 开始监听
channel.start_consuming()

# 关闭连接
cur.close()
conn.close()
```