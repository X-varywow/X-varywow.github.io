



## 1.Data cluster & micro-partitions

所有 snowflake tables 都会被自动划分，每个 micro-partition 对应 50～500 mb 未压缩的数据，同时具有一些元信息，方便后续的查询修剪（避免不必要的扫描，提高性能）。

对于经常查询且不经常更改的表，集群通常最具成本效益。

```sql
-- 1.cluster by base columns
create or replace table t1 (c1 date, c2 string, c3 number) cluster by (c1, c2);

-- 2.cluster by expressions
create or replace table t2 (c1 timestamp, c2 string, c3 number) cluster by (to_date(c1), substring(c2, 0, 10));

-- 3.cluster by paths in variant columns
create or replace table t3 (t timestamp, v variant) cluster by (v:"Data":id::number);

-- 4. Changing the Clustering Key
alter table t1 cluster by (c1, c3);

-- 5. drop the clustering key
alter table t1 drop clustering key;

-- 6. 显示 t1 的各种元信息：name,database,rows,cluster-by 等
show tables like 't1';
```

Data Warehouse Snowflake 最佳实践 中还有关于clustering key 的选取说明

参考官方文档：https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions.html

## 2.profile 性能分析

参考官方文档：[Analyzing Queries Using Query Profile](https://docs.snowflake.com/en/user-guide/ui-query-profile)


点击 view query profile 即可；或者根据 query_id 查询

以下是一些常见的性能开销：

| Data Access and Generation | Data Processing  | DML(data manipulation language) |
| -------------------------- | ---------------- | ------------------------------- |
| TableScan                  | Filter           | Insert                          |
| ValuesClause               | Join             | Delete                          |
| Generator                  | Aggregate        | Update                          |
| ExternalScan               | GroupingSets     | Merge                           |
| InternalObject             | WindowFunction   | Unload                          |
|                            | Sort             |                                 |
|                            | SortWithLimit    |                                 |
|                            | Flatten          |                                 |
|                            | JoinFilter       |                                 |
|                            | UnionAll         |                                 |
|                            | ExternalFunction |                                 |



以下是 Query Profile 可以发现的常见问题：

1. Exploding joins
通常表现为 join 不加 condition 或 conndition 不够导致join匹配的过多

2. Union without all
Union 相对于 unionall，除了简单的连接外，还会执行重复消除。unionall 够用时
应该使用 unionall, 不会有一个额外的 Aggregate 开销

3. Queries too large to fit in memory
As a result, the query processing engine will start spilling the data to local disk. If the local disk space is not sufficient, the spilled data is then saved to remote disks.
This spilling can have a profound effect on query performance (especially if remote disk is used for spilling). 
查询请求太大时，不再是常量级开销相较于小的查询。这其中涉及到 disk io, memory, data spilling 等开销
we recommend:
- Using a larger warehouse (effectively increasing the available memory/local disk space for the operation), and/or
- Processing data in smaller batches.

4. Inefficient pruning
低效的剪枝，当 filter 时没有太大变化，应该思考另一种数据组织方式或剪枝方式。
这体现在 tablescan 开销 上





## 3.Pipeline & Task

参考 https://docs.snowflake.com/en/sql-reference/ddl-pipeline

主要是 stream  和 task，定时执行 task 利用 stream 中的数据

CREATE TASK: https://docs.snowflake.com/en/sql-reference/sql/create-task




```sql
create task t1
    warehouse = ''
    schedule = ''
as
...

create task t2
    after t1
as
...

alter task t1 resume;
alter task t2 suspend;

execute task t1;
```



```sql
-- 1. 创建 task
create or replace task task1
	warehouse = house1
  -- 执行时间为上海的 8-21 点，每分钟执行一次
	schedule ='USING CRON */1 8-21 * * * Asia/Shanghai'
	USER_TASK_TIMEOUT_MS = 86400000
  -- 用于执行存储过程
	as call procedure1();


-- 2.创建过程
CREATE OR REPLACE PROCEDURE p_marketing_core_01_tmp   ()
RETURNS varchar    --返回值类型
LANGUAGE sql
AS
$$
    declare    --定义变量
    flag int;
begin
    flag := 1;  --变量赋值
----过程体----

  drop table  if exists _STAGE_RISK.risk_user_withdraw_dt_info_v1;
  create table  if not exists _STAGE_RISK.risk_user_withdraw_dt_info_v1
  as 
  select  user_id,source_flag
  from ODS_ASSETS.fact_withdraw_receipt_v1   
  group by  user_id,source_flag;

--------
  return flag;
end;
$$
;


-- 3 task任务查询

--3.1.启用task
ALTER TASK t_marketing_core_01_tmp resume; 

-- 3.2.停止task 
ALTER TASK t_marketing_core_01_tmp suspend; 

--3.3手动执行task
execute task t_marketing_core_01_tmp;

--3.4.删除task
drop task t_marketing_core_01_tmp;

--3.5.查看task
SHOW TASKS;

--3.6.查看task执行记录
//待补充
```

</br>

查看 TASK 执行记录：https://docs.snowflake.com/en/sql-reference/functions/task_history

```sql
select 
    concat(STATE,' : ',SCHEDULED_TIME) detail,
    datediff(second, QUERY_START_TIME, COMPLETED_TIME) cost_time,
    *
from table(
    information_schema.task_history(
        scheduled_time_range_start=>dateadd('hour',-24, current_timestamp()),
        result_limit => 3000,
        task_name=>'...'))
where SCHEMA_NAME = '...' and  DATABASE_NAME = '...';
```


!> task create or replace 之后要 resume


```sql
-- 检查 schedule， state 是 started 就行
SHOW TASKS LIKE 'task_name';

show tasks like 'line%' in warehouse.scheme;

-- % 为 通配符
-- 匹配以 tmp 为子串的字符串
SHOW TASKS LIKE '%tmp%';
```





## 4.Stream

用于记录一张表上的变化；

参考：
- https://docs.snowflake.com/en/user-guide/streams-examples
- https://docs.snowflake.com/en/sql-reference/sql/create-stream


```sql
create table MYTABLE1 (id int);

create table MYTABLE2(id int);

create stream MYSTREAM on table MYTABLE1;

insert into MYTABLE1 values (1);
-- returns true because the stream contains change tracking information
select system$stream_has_data('MYSTREAM');

--+----------------------------------------+
| --  | SYSTEM$STREAM_HAS_DATA('MYSTREAM') |
| --- | ---------------------------------- |
| --  | True                               |
--+----------------------------------------+

-- consume the stream
begin;
insert into MYTABLE2 select id from MYSTREAM;
commit;
-- returns false because the stream was consumed

select system$stream_has_data('MYSTREAM');
--+----------------------------------------+
| --  | SYSTEM$STREAM_HAS_DATA('MYSTREAM') |
| --- | ---------------------------------- |
| --  | False                              |
--+----------------------------------------+
```

<p class = "pyellow">这里 insert into target_table select 才能 consume 这个 stream，普通的 select 不行；</p>

```sql
show streams like 't2%';

drop stream t2;

-- 不存在不会报错
drop stream if exists t2;
```



简版流程：

```sql
create stream stream1 on table table1 append_only = true;

create or replace transient table tmp1
as

with cte as (
    select *
    from stream1
    where metadata$action = 'INSERT'
)
select * from cte;

merge into table2 target_table using
(select * from tmp1) source_table
on target_table.id = source_table.id
when matched then
    update set
when not matched then
    insert values ()
```



https://docs.snowflake.com/en/user-guide/streams-examples


双流数据 join 的问题，可以改为在原始表建立 join 的视图，然后使用 stream 捕获。（2024-02-22, 速度奇慢，基本无法用）



## 5.Dynamic Tables


一些比较复杂的问题，如双流中的数据关联起来才能使用，

https://docs.snowflake.com/en/user-guide/dynamic-tables-about


```sql
-- Create a landing table to store raw JSON data.
CREATE OR REPLACE TABLE raw(var VARIANT);

-- Create a stream to capture inserts to the landing table.
CREATE OR REPLACE STREAM rawstream1 ON TABLE raw;

-- Create a table that stores the names of office visitors from the raw data.
CREATE OR REPLACE TABLE names
(id INT,
first_name STRING,
last_name STRING);

-- Create a task that inserts new name records from the rawstream1 stream into the names table.
-- Execute the task every minute when the stream contains records.
CREATE OR REPLACE TASK raw_to_names
    WAREHOUSE = mywh
    SCHEDULE = '1 minute'
WHEN
    SYSTEM$STREAM_HAS_DATA('rawstream1')
AS
    MERGE INTO names n
    USING (
    SELECT var:id id, var:fname fname,
    var:lname lname FROM rawstream1
    ) r1 ON n.id = TO_NUMBER(r1.id)

    WHEN MATCHED AND metadata$action = 'DELETE' THEN DELETE

    WHEN MATCHED AND metadata$action = 'INSERT' THEN
    UPDATE SET n.first_name = r1.fname, n.last_name = r1.lname

    WHEN NOT MATCHED AND metadata$action = 'INSERT' THEN
    INSERT (id, first_name, last_name)
    VALUES (r1.id, r1.fname, r1.lname);

```

使用动态表后：

```sql
-- Create a landing table to store raw JSON data.
CREATE OR REPLACE TABLE raw(var VARIANT);

-- Create a dynamic table containing the names of office visitors from the raw data.
-- Try to keep the data up to date within 1 minute of real time.
CREATE OR REPLACE DYNAMIC TABLE names
    TARGET_LAG = '1 minute'
    WAREHOUSE = mywh
AS
SELECT 
    var:id::int id, 
    var:fname::string first_name,
    var:lname::string last_name 
FROM raw;
```










## 6.CDC

CDC, change data capture。指识别和捕获对数据库中的数据所做的更改，然后将这些更改实时交付给下游流程或系统的过程。

数据捕获的方式：
- 基于日志 （如 snowflake 中的 stream），**这是实施 CDC 的最有效的方式**
- 基于查询
- 基于触发器


## 7.External_functions

https://docs.snowflake.com/en/sql-reference/external-functions

https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-creating

snowflake 中使用 python：

```sql
-- 需要指定参数类型
create or replace function score_smooth_v1(scores array)
    returns double
    language python
    runtime_version = '3.8'
    comment = ''
    handler = 'score_smooth'
as
$$
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro


def score_smooth(scores):
    # 检验正态性
    shapiro_test = shapiro(scores)
    return shapiro_test
$$;


select score_smooth_v1([23, 151, 66, 46, 8, 8, 3, 101, 46, 62, 1, 175, 89, 12, 10, 10, 18, 37, 28, 17]);

```


## 8.UDTF

`UDF` user defined function

`UDTF` user defined table function


https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-tabular-vectorized


```sql
create or replace function stock_sale_sum(symbol varchar, quantity number, price number(10,2))
returns table (symbol varchar, total number(10,2))
language python
runtime_version=3.8
handler='StockSaleSum'
as $$
class StockSaleSum:
    def __init__(self):
        self._cost_total = 0
        self._symbol = ""

    def process(self, symbol, quantity, price):
      self._symbol = symbol
      cost = quantity * price
      self._cost_total += cost
      yield (symbol, cost)

    def end_partition(self):
      yield (self._symbol, self._cost_total)
$$;
```

```sql
select stock_sale_sum.symbol, total
from stocks_table, 
table(stock_sale_sum(symbol, quantity, price) over (partition by symbol));
```

















## 9.regexp

https://docs.snowflake.com/en/sql-reference/functions/regexp_substr

demo:

```sql
select 
    -- 提取 num= 后面的数字
    regexp_substr(context:"*args", 'num=([0-9]+)', 1, 1, 'e') as num,

    -- 贪婪匹配 [ ] 的内容
    regexp_substr(context:"result", '\\\\[.*?\\\\]') as ll
from t1
```



## 10.snowflake-scripting

[snowflake-scripting](https://docs.snowflake.com/en/developer-guide/snowflake-scripting/index)

```sql
DECLARE
  res RESULTSET;
  col_name VARCHAR;
  select_statement VARCHAR;
BEGIN -- begin end make block
  col_name := 'col1';
  select_statement := 'SELECT ' || col_name || ' FROM mytable';
  res := (EXECUTE IMMEDIATE :select_statement);
  RETURN TABLE(res);
END;
```





## 11.credits

[Understanding Compute Cost](https://docs.snowflake.com/en/user-guide/cost-understanding-compute)

Small (2 credits/hour) 

```sql
SELECT 
  query_id,
  name,
  scheduled_time,
  *
FROM 
  table(information_schema.task_history(scheduled_time_range_start=> dateadd('hours',-10,current_timestamp)))
WHERE schema_name = '' and database_name = ''
LIMIT 100;
```