
## _preface_

缓存：高性能、高并发

redis 记录在内存中，用作缓存

citus 常用于处理复杂查询和多种数据表关系


</br>

## _python redis_

https://www.runoob.com/w3cnote/python-redis-intro.html


```python
import redis

r = redis.Redis(host='localhost', port=6379, db = 0, decode_responses=True)   
r.set('name', 'runoob')
print(r['name'])
print(r.get('name'))
print(type(r.get('name')))
```


遍历所有 key:

```python
from redis import Redis

# 连接到 Redis
redis_client = redis.Redis(
    host="your_host", 
    port=6362)

# 定义要匹配的键的前缀
prefix = "game:"

# 使用 scan_iter 遍历匹配的键
for key in redis_client.scan_iter(f"{prefix}*"): # f"{prefix}*"
    print(key)
    # 在这里可以执行其他操作，比如获取哈希值的内容
#     hash_value = redis_client.hgetall(key)
#     print(hash_value)
```



| 命令 | 说明                 |
| ---- | -------------------- |
| scan | 一个基于游标的迭代器 |
|      |                      |
|      |                      |

----------

```bash
SCAN 0 MATCH "bike:*" COUNT 100
```

SCAN returns a cursor position, allowing you to scan iteratively for the next batch of keys until you reach the cursor value 0.




</br>

## _集群模式_













</br>

## _Redis 面试_

参考：
- [面试官：你对Redis缓存了解吗？面对这11道面试题你是否有很多问号？](https://zhuanlan.zhihu.com/p/136796077)

>(1) 了解 Redis 吗

Redis 是一个开源的、C实现的、高性能的、基于内存的、单线程的Key_value系统。

适用于高QPS、低延迟、若持久化的场景，用作缓存。

>(2) redis 支持的数据类型？

- String 字符串（做简单的KV操作）
- Hash 散列（类似 map 的一种结构，将一个结构化的数据缓存给 redis）
- List 列表
- Set 集合（自动去重）
- Sorted Set 有序集合

>(3) redis 持久化

持久化，即将数据写入内存的同时，异步地将数据写入磁盘文件里。主要用于灾难恢复、数据恢复。

持久化的两种机制：
- RDB(Redis DataBase)，对 redis 中的数据执行周期性的保存
- AOF(Append Only File)，将每条写入命令作为日志，以 append-only 的模式写入一个日志文件中，在 redis 重启的时候，可以通过回放 AOF 日志中的写入指令来重新构建整个数据集

>(4) 常见的缓存问题

缓存用于：加快页面打开速度，减少网络带宽消耗，降低服务器压力

>(5) redis 和 memcached 有什么区别？redis 的线程模型是什么？

redis 支持更加复杂的数据结构，redis 原生支持集群模式。redis只使用单核，而memcached可以使用多核，平均每一个核上redis在存储小数据时比memcached性能更高。而在100k以上数据中，memcached性能要高于redis.

>(6) 为什么 redis 单线程却能支撑高并发？Redis快速原因？

- redis 是基于内存的，内存的读写速度比磁盘快很多
- redis 是单线程的，省去了许多上下文切换的时间
- redis 使用了epoll多路复用技术，可以处理并发的连接


>(7) 了解什么是 redis 的雪崩、穿透和击穿？redis 崩溃之后会 怎么样？系统该如何应对这种情况？如何处理 redis 的穿透？

缓存雪崩和穿透是缓存最大的两个问题，一旦出现就是致命性的问题。

**缓存雪崩**：假设服务器每日高峰期会有5000个请求，缓存能抗住4000个。然后缓存机器宕机了，这时所有请求落在数据库系统上了，他会报一下警，然后挂掉。重启也会被新的流量撑死。

缓存雪崩的事前事中事后的解决方案如下。 
- 事前：redis 高可用，主从+哨兵，redis cluster，避免全盘崩溃。 
- 事中：本地 ehcache 缓存 + hystrix 限流&降级，避免 MySQL 被打死。 
- 事后：redis持久化，一旦重启，自动从磁盘上加载数据，快速恢复缓存数据。

**缓存穿透**：即服务器收到很多的不在缓存中，也不在数据库中的请求，比如黑客的恶意攻击，这时大量的请求会影响数据库系统的稳定性。

解决方法：将没查到写一个空值到缓存里去，如：set -999 UNKNOWN，然后设置一个过期时间

**缓存击穿**：某个key值的访问非常频繁，在这个key失效时，大量的缓存就击穿了缓存，直接请求数据库。

解决方法：将热点数据设置为永不过期；或基于 redis or zookeeper 实现互斥锁

>(8) 了解 redis 集群吗？

Redis 集群是 Redis 提供的**分布式数据库方案**，集群通过分片来实现数据共享，并提供复制和故障转移。

Redis 支持三种集群方案：
- 主从复制模式
- Sential(哨兵)模式
- Cluster 模式

参考：[Redis 集群](https://cloud.tencent.com/developer/article/1592432)

>(9) 数据不一致问题

一般的业务场景都是读多写少的，当客户端的请求太多，对数据库的压力越来越大，引入缓存来降低数据库的压力是必然选择，目前业内主流的选择基本是使用 Redis 作为数据库的缓存。但是引入缓存之后，可能出现缓存与数据库的数据一致性问题。

根据不同的业务，不同的数据一致性要求，结合系统的性能综合考虑，选择适合自己系统的方案就好。
- 先操作 Redis，再操作数据库
- 先操作数据库，再操作 Redis

总的来说，由于我们的基本原则是以数据库为准，那么我们选择的方案就应该把操作数据库放到前面，也就是说我们应该先操作数据库，再操作 Redis，对于并发很高的场景，我们可以在操作数据库之前通过消息队列来降低客户端对数据库的请求压力。

参考：[Redis 的数据一致性方案分析](https://zhuanlan.zhihu.com/p/141537171)
