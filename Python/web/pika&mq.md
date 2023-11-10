

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