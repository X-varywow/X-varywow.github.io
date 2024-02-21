
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


>**B树索引** 用于组合索引时，一旦某个字段使用了范围查询，则该字段后的所有索引字段将不会依序使用 B树索引进行查找优化。</br></br>
>通常将等值查询的条件放在索引字段的左边，将范围查询放在最右边，从而最大化索引的效用。</br></br>
>如 组合索引（A,B,C）,合理的查询顺序是 A=x AND B=y AND C>z

[索引失效的10种场景](https://zhuanlan.zhihu.com/p/455188214)


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






---------

参考资料：
- [psql 中文文档](https://docs.postgresql.tw/)
- [psql 菜鸟教程](https://www.runoob.com/postgresql/postgresql-tutorial.html)
- chatgpt
