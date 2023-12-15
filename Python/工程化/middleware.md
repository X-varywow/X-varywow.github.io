
位于应用程序的核心逻辑和底层系统之间，用于处理请求、响应和数据流转等中间步骤。

- FIFO cache
- LFU cache (least frequently used)
- LRU cache (least recently used)
- MRU cache (most recently used)
- RR cache (random replacement)
- TTL cache
- TLRU cache (Time-aware LRU)



## _cachetools_

缓存是一种用于临时存储计算结果的技术，以避免在后续计算中重复执行相同的计算。使用缓存可以提高应用程序的性能和响应速度。

```python
import cachetools

# 最大键值对为 500， 超时时间为 0.2s
ttl_cache = cachetools.TTLCache(maxsize=1024, ttl=0.2)

cache_k = f"{uid}"
v = ttl_cache.get(cache_k)
if v:
    var = v

ttl_cache[cache_k] = new_v
```

ttl_cache 通过设置一个时间点，然后 <u> 轮询检查过期时间与当前时间的关系 </u>


-------------

自定义缓存策略：

```python
import cachetools

class MyCache(cachetools.Cache):
    def __init__(self, maxsize):
        super().__init__(maxsize = maxsize)
    
    def __getitem__(self, key, cache_getitem = dict.__getitem__):
        return cache_getitem(self, key)

    def __setitem__(self, key, value, cache_setitem = dict.__setitem__):
        if len(self) >= self.maxsize:
            self.popitem(last=False)
        cache_setitem(self, key, value)

cache = MyCache(maxsize=100)
```



</br>

## _pika_



```python
import pika
import psycopg

# 连接到消息队列
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 创建队列
channel.queue_declare(queue='my_queue')

# 连接到PostgreSQL数据库
conn = psycopg.connect(database="your_database", user="your_user", password="your_password", host="your_host", port="your_port")
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

</br>

## _kafka-python_


https://huzzefakhan.medium.com/apache-kafka-in-python-d7489b139384



--------------

参考资料：
- [cachetools库简介以及详细使用](https://developer.aliyun.com/article/1207758)