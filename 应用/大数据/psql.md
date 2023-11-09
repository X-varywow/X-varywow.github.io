
## Preface

PostgreSQL 是一个免费的对象-关系数据库服务器，本页面只介绍 psql 特殊语法

SQL COMMAND，[请参考](cs/DATABASE/base)

## 连接

- 命令行方式
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
| set search_path to path_name |                           |
|                              |                           |






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
```


## 建表规范



```sql
-- 创建表 + 主键索引
create table "scheme_name"."table_name"(
  "id" int8 not null
  "col2" int2,
  "json_type" jsonb,
  constraint "pkey_name" primary key ("id", "col2")
);

-- 二级索引
create index "index_name" on "table_name" using btree ("class", "student_name");



-- 列注释
comment on column "scheme_name"."table_name"."col_name" is 'introduction';

-- 分片
set citus.shard_count = 32;
select create_distributed_table('table_name', 'col_name');
```

> 分片可以将整体的数据集划分为多个部分。这是为了解决大规模数据存储和处理的问题，常用于分布式数据库和分布式存储系统中。


---------

参考资料：
- [psql 中文文档](https://docs.postgresql.tw/)
- [psql 菜鸟教程](https://www.runoob.com/postgresql/postgresql-tutorial.html)
