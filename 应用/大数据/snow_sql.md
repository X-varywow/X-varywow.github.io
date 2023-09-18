## 一、基础sql

数据库相关，[请参考](/cs/DATABASE/)

## 二、基础语法


```sql
-- query procedure...

select get_ddl("procedure", procedure_name);

select get_ddl("table", table_name);

```

### 2.1 控制流

#### 2.1.1 使用 递归

```sql
with recursive t(n) as
(
    select 1
    union all
    select n+1 from t
)
select n from t limit 10
```

#### 2.1.2 script block

```sql
Begin end 定义了一个 script block
execute immediate $$
begin
    create table parent (id integer);
    create table child (id integer, parent_id integer);
    return 'Completed';
end;
$$
;
```


#### 2.1.3 使用while循环、break

```sql
declare
    i integer;
    j integer;
begin
    i := 1;
    j := 1;
    while (i <= 4) do
        while (j <= 4) do
        -- Exit when j is 3, even if i is still 1.
            if (j = 3) then
                break outer_loop;
            end if;
            j := j + 1;
         end while inner_loop;
         i := i + 1;
     end while outer_loop;
     -- Execution resumes here after the BREAK executes.
     return i;
 end;
```

#### 2.1.4 使用 case

```sql
create procedure case_demo_01(v varchar)
returns varchar
language sql
as
begin
    case (v)
        when 'first choice' then
            return 'one';
        when 'second choice' then
            return 'two';
        else
            return 'unexpected choice';
    end;
end;
```

#### 2.1.5 使用 loop

```sql
create table dummy_data (id integer);

create procedure break_out_of_loop()
returns integer
language sql
as
$$
    declare
        counter integer;
    begin
        counter := 0;
        loop
            counter := counter + 1;
            if (counter > 5) then
                break;
            end if;
            insert into dummy_data (id) values (:counter);
        end loop;
        return counter;
    end;
$$
;
```

#### 2.1.6 使用 for

```sql
-- 1. 基于计数器

for i in 1 to iteration_limit do
    。。。
end for;

-- 2. 基于游标

create or replace procedure for_loop_over_cursor()
returns float
language sql
as
$$
declare
    total_price float;
    c1 cursor for select price from invoices;
begin
    total_price := 0.0;
    open c1;
    for rec in c1 do
        total_price := total_price + rec.price;
    end for;
    close c1;
    return total_price;
end;
$$
;
```


#### 2.1.7 使用 if

```sql
if (flag = 1) then
    return 'one';
elseif (flag = 2) then
    return 'two';
else
    return 'Unexpected input.';
end if;
```


### 2.2 特殊

#### 2.2.1 使用变量和游标


Let 使用变量：
https://docs.snowflake.com/en/sql-reference/snowflake-scripting/let.html


使用游标：
https://docs.snowflake.com/en/developer-guide/snowflake-scripting/cursors.html




#### 2.2.2 窗口相关

1. ratio_to_report
将当前行中的值除以窗口中所有行中的值的总和。这是前面查询的等效项。???

```sql
select branch_id,
    city,
    100 * ratio_to_report(net_profit) over (partition by city)
from store_sales as s1
order by city, branch_id;
```


2. 使用 percentile_cont

```sql
create or replace table aggr(k int, v decimal(10, 2));

insert into aggr(k, v) values
    (0, 1),
    (0, 2),
    (1, 2);

select k, percentile_cont(0.5) within group(order by v)
from aggr
group by k
order by k;

-- 每个k对应着一个数组，如0:【1，2】，1:【2】，都会返回 0.5 分位点的数
--      k    percentile_cont(0.5) within group(order by v)
-- 0    0                  1.5
-- 1.   1                   2
```

3. 使用 row_number()

Returns a unique row number for each row within a window partition.

```sql
select state, bushels_produced, row_number()
over (order by bushels_produced desc)
from corn_production;
+--------+------------------+------------+
|  state | bushels_produced | ROW_NUMBER |
+--------+------------------+------------+
| Kansas |              130 |           1|
| Kansas |              120 |           2|
| Iowa   |              110 |           3|
| Iowa   |              100 |           4|
+--------+------------------+------------+
select 
symbol,
exchange,
shares,
row_number() over (partition by exchange order by shares) as row_number
from trades;
+------+--------+------+----------+
|SYMBOL|EXCHANGE|SHARES|ROW_NUMBER|
+------+--------+------+----------+
|SPY   |C       |   250|         1|
|AAPL  |C       |   250|         2|
|AAPL  |C       |   300|         3|
|SPY   |N       |   100|         1|
|AAPL  |N       |   300|         2|
|SPY   |N       |   500|         3|
|QQQ   |N       |   800|         4|
|QQQ   |N       |  2000|         5|
|YHOO  |N       |  5000|         6|
+------+--------+------+----------+
```

>- row_number() 不会跳过 rank，重复时为不同 rank
>- dense_rank() 不会跳过 rank，重复时为同 rank
>- rank() 会跳过 rank，重复时为同 rank


4. 使用lag

同一表中不同行

```sql
SELECT emp_id, year, revenue, 
       revenue - LAG(revenue, 1, 0) OVER (PARTITION BY emp_id ORDER BY year) AS diff_to_prev 
FROM sales 
ORDER BY emp_id, year;
```


5. 使用 QUALIFY

QUALIFY 对窗口函数的作用就像 HAVING 对聚合函数和 GROUP BY 子句的作用一样。

```sql
-- 筛选记录数 >=10 的所有用户的所有记录
select * from table_name
where
qualify count(1) over (partition by user_id) >= 10
```




#### 2.2.3 function & procedure


1. 使用简易的 function

```sql
create or replace function get_rows()
returns number as 'select count(*) from table_name';

select get_rows();
```

2. 使用简易的 procedure

```sql
create or replace procedure count_rows()
returns number 
language sql
as 
begin 
 return (select count(*) from table_name);
end;

call count_rows();
```


!> Function 和 procedure 有利于增强代码复用性，区别：函数一般情况下用来计算并返回一个计算结果。存储过程一般是用来完成特定的操作，sql语句（DML或select）中不可调用。

3. Procedure 返回表格数据

```sql
create or replace procedure get_entry_fee() 
returns table() 
language sql
as 
declare
 res resultset default(
    select
        user_id,
    from
        table_name
    GROUP BY
        USER_ID,
        DT
    limit
        10);
begin
    return table(res);
end;

call get_entry_fee();
```


4. 使用参数

```sql
create or replace procedure get_entry_fee(source_flag int, cnt int) 
returns table() 
language sql
as 
declare
 res resultset default(
    select
        user_id,
        DT,
        ROUND(avg(actual_entry_fee), 2) avg_entryfee,
        ROUND(min(actual_entry_fee), 2) min_entryfee,
        ROUND(max(actual_entry_fee), 2) max_entryfee
    from
        table_name
    GROUP BY
        USER_ID,
        DT
    limit 10 );
begin
    return table(res);
end;

call get_entry_fee(2, 5);
```

#### 2.2.4 动态构建 SQL 语句

```sql
CREATE OR REPLACE PROCEDURE test_sp_dynamic(table_name VARCHAR)
RETURNS TABLE(a INTEGER)
LANGUAGE SQL
AS
    DECLARE
        res RESULTSET;
        query VARCHAR DEFAULT 'SELECT a FROM ' || :table_name || ' ORDER BY a';
    BEGIN
        res := (EXECUTE IMMEDIATE :query);
        RETURN TABLE (res);
    END;

-- 传入表名，调用该存储过程即可
call test_sp_dynamic('t001');
```


#### 2.2.5 其他

- 创建临时表
- 加载数据 https://docs.snowflake.com/en/user-guide/data-load-overview.html
- array_agg() 将一列输出转化为数组，https://docs.snowflake.com/en/sql-reference/functions/array_agg.html


## 三、高级应用

### 3.1 Data cluster & micro-partitions

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

### 3.2 profile 性能分析

参考官方文档：Analyzing Queries Using Query Profile


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



### 3.3 Snowpark


参考官方文档：https://docs.snowflake.com/en/developer-guide/snowpark/index.html
snowpark提供了一个直观的 API，用于查询和处理数据管道中的数据。
使用此库，您可以构建在 Snowflake 中处理数据的应用程序，而无需将数据移动到运行应用程序代码的系统。

相比 snowflake connector，snowpark优势：
- supports pushdown for all operations, including Snowflake UDFs
- not require a separate cluster outside of Snowflake for computations. All of the computations are done within Snowflake.

具体操作请参考：
- 适用于 Python 的 Snowpark 开发人员指南
- Snowpark Python: Bringing Enterprise-Grade Python Innovation to the Data Cloud

### 3.4 Pipeline & task

参考 https://docs.snowflake.com/en/sql-reference/ddl-pipeline

主要是 stream  和 task，定时执行 task 利用 stream 中的数据

CREATE TASK: https://docs.snowflake.com/en/sql-reference/sql/create-task



```sql
show tasks like 'line%' in warehouse.scheme;
```





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
```



```sql

-- 1.创建 task 由于需具有创建task的权限
CREATE OR REPLACE TASK t_marketing_core_01_tmp
  WAREHOUSE = PRODUCTION_AVIA_DEFAULT_MEDIUM_1
--task执行的北京时间 给个30分钟   30*60 = 1800 000
  SCHEDULE = 'USING CRON 20 07 * * * Asia/Shanghai'
  USER_TASK_TIMEOUT_MS = 1800000
AS
--需要执行的存储过程
  CALL p_marketing_core_01_tmp();


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












### 3.5 Stream

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








### 3.8 Dynamic Tables


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















### 3.7 CDC

CDC, change data capture。指识别和捕获对数据库中的数据所做的更改，然后将这些更改实时交付给下游流程或系统的过程。

数据捕获的方式：
- 基于日志 （如 snowflake 中的 stream），**这是实施 CDC 的最有效的方式**
- 基于查询
- 基于触发器


### 3.8 External_functions

https://docs.snowflake.com/en/sql-reference/external-functions

https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-creating


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


### 3.9 UDTF


`UDF` user defined function

`UDTF` user defined table function


https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-tabular-vectorized










## 四、other

(1) 通过临时表减少实际需要的中间表

```sql
with cte1 as (
-- 这里语句结束不加 ;
),
cte2 as (
  select *
  from cte1
) 
-- 这里不加, 后面接着 select
select *
from cte2;
```

(2) 构建 json 数据以传输

```sql
select object_construct(*) as json_row from (
    select *
    from table_name
) t;
```
```sql
insert overwrite into table_name
select 
       object_construct(
        'col1', 2,
        'col2', 3,
       )
from t2;
```


(3) 使用 MERGE INTO 将源表合并到目标表
```sql
MERGE INTO target_table t
USING source_table s
ON t.id = s.id
WHEN MATCHED THEN
  DELETE;
```



(4) 类型转换
```sql
select to_varchar(2);

select '2';
```

