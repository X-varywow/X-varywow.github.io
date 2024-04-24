
## Preface

PostgreSQL 是一个免费的对象-关系数据库服务器，本页面只介绍 psql 特殊语法

SQL COMMAND，[请参考](cs/DATABASE/base)





## 连接

- 命令行方式(psql)
- GUI 方式
  - pgadmin
  - navicat

## 命令行

```bash
/usr/local/opt/postgresql@14/bin/psql --help

/usr/local/opt/postgresql@14/bin/psql -h your_host -p your_port -U your_user -d your_database
```

| 命令                         | 说明                      |
| ---------------------------- | ------------------------- |
| \l                           | list of databases         |
| \d                           | list of relations, schema |
| \d table_name                |                           |
| \d+ table_name               | 列出表的信息              |
| \c database_name             | connected to database     |
| \du                          | list of users             |
| \dn                          | list of schemas           |
| set search_path to path_name |                           |
|                              |                           |



## 建表规范



```sql
-- 创建表 + 主键索引
create table scheme_name.table_name(
  "id"        int not null
  "col2"      bigint,
  "json_type" jsonb,
  created_at   timestamp default current_timestamp
  constraint "pkey_name" primary key ("id", "col2")
);

-- 二级索引
create index "index_name" on "table_name" using btree ("class", "student_name");


-- 列注释
comment on column "scheme_name"."table_name"."col_name" is 'introduction';

-- 分片
set citus.shard_count = 32;
select create_distributed_table('schema.table_name', 'col_name');

-- 参照表（保证每个节点都有完整的数据副本，适用于数据量不大但需经常访问的表）
SELECT create_reference_table('schema.table_name');

-- 分区
create table ...()
partition by range(created_at);

select create_time_partitions(
    table_name := 't1',
    partition_interval := '1 month',
    start_from := '2023-09-01 00:00:00',
    end_at := now() + '24 months'
)
```

分片可以将整体的数据集划分为多个部分。这是为了解决大规模数据存储和处理的问题，常用于分布式数据库和分布式存储系统中。

set citus.shard_count = 32; 意味着 64 个分片，可以分布在不同的物理服务器节点上。

`缺点`

增加分片数量，可能增加连接数量，因为客户端或 coordinator 节点可能需要与更多的工作节点通讯

分片数目太少可能导致数据分布不均，限制了扩展性和并发能力；

分片数过多导致开销上升


> psql 的分区是指一张大表被水平切分为多份小的、独立的表；citus 的分片与 psql 的分区在实现方式上有所不同；</br>
> 一张表是否是分区表取决于它是否在citus集群外已经被定义为分区表，即使 citus 设置成分片 32，内部也可没有分区。



## psql 语法

条件表达式，[官方文档](https://docs.postgresql.tw/the-sql-language/functions-and-operators/conditional-expressions)

[postgresql中在查询结果中将字符串转换为整形或浮点型](https://blog.csdn.net/qq_40323256/article/details/124292446)

[postgresql将字段为空的值替换为指定值](https://www.jianshu.com/p/bf0101f06535)


coalesce(arg1, arg2, ...) 返回第一个 不为null的值


psql 取 json 中的数据：(json_row->>'user_id') 





```sql
-- 生成10个 0 到 2000 的随机数
SELECT floor(random() * 2001) AS random_number
FROM generate_series(1, 10) AS s;

-- 校验
select max(random_number), min(random_number)
from (
	SELECT floor(random() * 3)+1 AS random_number
	FROM generate_series(1, 10000) AS fo
) as foo;


-- 生成随机数并转化为字符串：
select CAST( (floor(random() * 3)+1)*10000 + floor(random() * 2001) AS VARCHAR);



-- 更新多列值
update t1
set c1 = {}, c2 = {}
where user_id = {};
```


```sql
-- 获取时间

select date_part('hour', t);
```


```sql
-- 新增字段
alter table t1 add column user_id bigint;

-- 修改字段
alter table t1 alter column col_name type text;

-- 重命名
alter table schema_name.t1 rename to t2;

-- 新增索引
create index "index_name" on "table_name" using btree ("class", "student_name");
```


## other

`SELECT * FROM t LIMIT 100;` 这条 SQL 语句的核心作用是从表 `t` 中选取前 100 行数据。其原理和对服务的影响主要取决于几个因素：

#### 原理
1. **解析查询：** 数据库首先解析SQL语句，理解需要执行的操作（在这里是选择全部列，从表 `t` 中）。
2. **计划和优化：** 接着，数据库会制定一个执行计划，可能会考虑是否使用索引（如果有的话），以及如何高效地获取数据。
3. **执行：** 按照执行计划，数据库读取表 `t` 的数据。如果表很大，但没有合适的索引来直接定位到前 100 行，数据库可能需要扫描大量的行。
4. **限制结果：** 尽管最终只返回前 100 行给用户，但是根据表的大小和配置，以及是否利用有效索引，数据库的工作量可能有很大差异。
5. **发送结果：** 最后，选取的数据被发送给请求者。

#### 对服务的影响
- 在很多情况下，使用 `LIMIT` 语句不会对服务造成显著的压力，尤其是对于优化良好且拥有合理索引的数据库。
- 如果表非常大，而且缺少有效的索引，数据库执行这一查询操作时会进行大量的数据读取和事务日志记录，这可能会产生较大的I/O压力和CPU使用增加，从而影响数据库的性能。
- 对于读取性能较差的存储媒介，或者当数据库承载的其他读写操作已经很重时，执行这样一个查询确实可能引起性能尖刺。在极端情况下，如果表 `t` 非常大且查询需要回退到磁盘读取大量数据（对于没有足够内存缓存所有数据的情况），性能问题更加明显。
- 使用 `SELECT *` 而不是指定特定的列可能会增加网络传输负担，尤其是当表含有大量列且部分列存储了大量数据（如文本或二进制数据）时。

**总的来说**，`SELECT * FROM t LIMIT 100;` 对性能的影响很大程度上取决于表的大小、表的索引配置以及数据库服务器的性能。对于大多数情况，尤其是小到中等数据集，和/或良好设计的数据库，在执行这样的查询时不太可能造成服务的显著负担。不过，为了优化性能，建议尽可能地使用索引，根据需要选择特定的列，并关注查询执行计划。

!> 更容易造成尖刺的是 ide 的 view data 按钮

---------


>**B树索引** 用于组合索引时，一旦某个字段使用了范围查询，则该字段后的所有索引字段将不会依序使用 B树索引进行查找优化。</br></br>
>通常将等值查询的条件放在索引字段的左边，将范围查询放在最右边，从而最大化索引的效用。</br></br>
>如 组合索引（A,B,C）,合理的查询顺序是 A=x AND B=y AND C>z

[索引失效的10种场景](https://zhuanlan.zhihu.com/p/455188214)



---------

参考资料：
- [psql 中文文档](https://docs.postgresql.tw/)
- [psql 菜鸟教程](https://www.runoob.com/postgresql/postgresql-tutorial.html)
- chatgpt
