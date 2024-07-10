


- 用于编写 **并发**，使用 async / await 语法
- 被用作多个提供高性能 Python 异步框架的基础，包括网络和网站服务，数据库连接库，分布式任务队列等等
- 往往是构建 IO 密集型和高层级 结构化 网络代码的最佳选择

优点：非阻塞性，在IO密集型任务可以显著提高效率

## 1. 协程与任务

```python
import asyncio


# 使用 async await 来声明协程
async def main():
    print('hello')
    await asynico.sleep(1)
    print('world') 


asyncio.run(main())

# 简单地调用协程并不会使其调度执行
main()
```


```python
import asyncio

async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    await asyncio.gather(count(), count(), count())

asyncio.run(main())

# 总共耗时约 1s 
# -> 1 1 1 2 2 2
```









```python
# 会一个一个执行
async def main():
    await func(1, "hello")
    await func(1, "hello")


# 使用 asyncio.create_task() 来并发运行协程
async def main():
    task1 = asyncio.create_task(func(1, "hello"))
    task2 = asyncio.create_task(func(1, "hello"))

    await task1
    await task2

asyncio.run(main())
```

> 可以在 await 中使用的，称为**可等待对象**。有：协程，任务，future

```python
# 使用 gather 来并发运行任务，其中的可等待对象为协程时，自动被作为一个任务调度
await asyncio.gather(
    func(1),
    func(2),
    func(3)
)
```

## 2. 流

流是用于处理网络连接的支持 async/await 的高层级原语。 流允许发送和接收数据，而不需要使用回调或低级协议和传输。

下面是一个使用 asyncio streams 编写的 TCP echo 客户端示例:

```python
import asyncio

async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 8888)

    print(f'Send: {message!r}')
    writer.write(message.encode())
    await writer.drain()

    data = await reader.read(100)
    print(f'Received: {data.decode()!r}')

    print('Close the connection')
    writer.close()
    await writer.wait_closed()

asyncio.run(tcp_echo_client('Hello World!'))
```

其他：（同步原语）（子进程 subprocess）


----------

参考资料：
- [官方文档](https://docs.python.org/zh-cn/3/library/asyncio.html?highlight=asyncio#module-asyncio)
- [Python 异步编程入门 - 阮一峰](https://www.ruanyifeng.com/blog/2019/11/python-asyncio.html)