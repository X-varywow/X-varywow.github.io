

citus 是 postgres 的开源扩展；使用 citus 功能：分片、分布式表、引用表、分布式查询引擎、列式存储；以及从 Citus 11.0 开始，从任何节点进行查询的能力。Citus 将并行性、在内存中保留更多数据和更高的 I/O 带宽相结合，可以显着提高多租户 SaaS 应用程序、面向客户的实时分析仪表板和时间序列工作负载的性能。


## 1. 基本概念


- 分布式表、分区表（很大的表，分片又分区）
  - create_distributed_table
- Reference 表（较小的表，在各个机器上都有一份完整数据）
  - create_reference_table
- 本地表
  - 没有经过分片的 postgresql 原生的表

citus 11.0 后所有数据集群使用原数据同步。任何节点都可以当做协调节点，即从任何节点运行分布式 psql 查询。

分片，数据库横向扩展到多个物理节点，每一个分区只包含部分数据

副本，


选择合适的表结构：https://docs.citusdata.com/en/stable/sharding/data_modeling.html


## 2. 优势

并行查询，允许跨多个节点查询，并通过向集群中添加新节点来提高处理速度；

另外，将单个查询分割成多个片段来增强并行性


## 3. 语句

```sql
-- 设置副本数
set citus.shard_replication_factor = 2;


-- 创建分布表
select create_distributed_table(table_name, column_name);


-- 分片
show citus.shard_count;


-- 查看分片
select * from pg_dist_shard;
select * from pg_dist_shard_placement;

-- 查看索引
select * from pg_indexes where tablename=""

-- 更改分片数为 2
select alter_distributed_table(table_name, shard_count=2, cascade_to_colocated:=true);

-- 查看分片位置
select get_shard_id_for_distribution_column(table_name, 1);

```


## 4. 应用场景

### 4.1 多租户数据库

https://docs.citusdata.com/en/stable/use_cases/multi_tenant.html


### 4.2 实时分析

https://docs.citusdata.com/en/stable/use_cases/realtime_analytics.html



```sql
CREATE TABLE github_events
(
    event_id bigint,
    event_type text,
    event_public boolean,
    repo_id bigint,
    payload jsonb,
    repo jsonb,
    user_id bigint,
    org jsonb,
    created_at timestamp
);

CREATE TABLE github_users
(
    user_id bigint,
    url text,
    login text,
    avatar_url text,
    gravatar_id text,
    display_login text
);

-- gin 索引可以更快地查询 jsonb 字段
CREATE INDEX event_type_index ON github_events (event_type);
CREATE INDEX payload_index ON github_events USING GIN (payload jsonb_path_ops);

-- 分片，Citus 将这些表分布在集群中的节点上
SELECT create_distributed_table('github_users', 'user_id');
SELECT create_distributed_table('github_events', 'user_id');
```


```sql
SELECT count(*) FROM github_users;

-- 每分钟推送时间提交数
SELECT date_trunc('minute', created_at) AS minute,
       sum((payload->>'distinct_size')::int) AS num_commits
FROM github_events
WHERE event_type = 'PushEvent'
GROUP BY minute
ORDER BY minute;
```

```sql
-- 过期旧数据
```


## 5. 性能优化


查看 sql 耗时情况：

`explain` 显示基本执行计划

`explain analyze` 显示基本执行计划 + 耗时情况

`explain verbose` 显示冗长信息

```python
sql = """
EXPLAIN
select * from t1 limit 100;
"""


with (
    psycopg.connect(PG_CONFIG) as conn,
    conn.cursor() as cur
):
    cur.execute(sql)
    res = cur.fetchall()
    for row in res:
        print(row[0])
```




参考 [citus 性能优化 官方文档](https://docs.citusdata.com/en/stable/performance/performance_tuning.html)




## 6. Advanced

`pgbouncer` 是一个用于PostgreSQL数据库连接池的轻量级 **连接池代理**。它允许多个客户端应用程序共享一个或多个数据库连接，从而有效地管理数据库连接并提高数据库性能。主要是连接池管理、负载均衡、故障转移

`Patroni` 是一个开源的PostgreSQL高可用性解决方案。它是使用Python编写的，基于ZooKeeper和etcd等分布式协调服务来提供自动故障转移和自动恢复的功能。

**主从复制架构**，其中一个节点被选为主节点，其余节点作为从节点。如果主节点宕机，Patroni会自动将其中一个从节点提升为新的主节点，以确保数据库的连续可用性。（这种思想与冗余还是有些异同，就叫主从吧）


Citus 12 新特性：基于 scheme 的分片，支持 PG16, 从任意节点查询时的负载均衡



----------

参考资料：
- https://www.bilibili.com/video/BV1h84y1B73E/
- [citus 官方文档](https://docs.citusdata.com/)
- chatgpt
