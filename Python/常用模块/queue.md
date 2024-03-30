

## _queue_

一个同步的队列类；

实现了多生产者，多消费者队列；特别适合于 <u>消息安全地在多线程间交换</u> 的线程编程

官方文档：https://docs.python.org/zh-cn/3/library/queue.html



</br>

## _语法_

```python
# qsize()>0 不保证后续 get() 不被阻塞
q.qsize()

q,empty()
q.full()

q.put()
q.put_nowait()

q.get()
q.get_nowait()


# 表示前面排队的任务已经被完成
q.task_done()

# 阻塞至队列中所有的元素都被接收和处理完毕
# 每当一个消费者线程调用 task_done() 来表明该条目已被提取且其上的所有工作已完成时未完成计数将会减少。 
q.join()

```





</br>

## _实例_

demo1：任务队列

```python
import threading
import queue
import time

q = queue.Queue()

def worker():
    while True:
        item = q.get()
        print(f"Working on {item}")
        time.sleep(0.5)
        print(f"Finished {item}")
        
        q.task_done()
    
# Turn-on the worker thread
threading.Thread(target=worker, daemon=True).start()

# Send task requests to the worker
for item in range(30):
    q.put(item)

# Block until all tasks are done
q.join()
print("All work completed")
```

demo2：(线程安全)多生产者、消费者


```python
import threading
import time
import queue

q = queue.Queue()

def producer(queue, id):
    for i in range(5):
        item = f"item_{id}_{i}"
        print(f"producer {id} produce {item}")
        queue.put(item)
        time.sleep(0.5)
    print(f"producer{id} complete")
    
def consumer(queue, id):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"consumer {id} consume {item}")
        queue.task_done()
        
producers = [threading.Thread(target = producer, args=(q,i)) for i in range(4)]
for p in producers:
    p.start()
    
consumers = [threading.Thread(target = consumer, args=(q,i)) for i in range(3)]
for c in consumers:
    c.start()
    
# 等待生产者结束
for p in producers:
    p.join()
# 发送终止信号
for i in range(len(consumers)):
    q.put(None)
#等待消费者结束
for c in consumers:
    c.join()
```


