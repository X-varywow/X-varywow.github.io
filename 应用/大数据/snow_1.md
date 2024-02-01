
## preface

Snowflake 是一个 云端数据仓库，完全在云基础设施上运行。

Snowflake 数据平台并非建立在任何现有的数据库技术或 Hadoop 等“大数据”软件平台之上。相反，Snowflake 将全新的 SQL 查询引擎与专为云原生设计的创新架构相结合。对于用户，Snowflake 提供了企业分析数据库的所有功能，以及许多额外的特殊功能和独特功能。


介绍1：[云数仓神话 Snowflake 的增长之路](https://mp.weixin.qq.com/s/wkaRlBxNEVEAJraecR-hiQ)


(snowflake 架构)，由三个基本层组成：
- Database Storage
- Query Processing
- Cloud Services


--------------

SQL BASIC COMMAND，请参考：[cs/DATABASE/base](cs/DATABASE/base)




```sql
-- query procedure...

select get_ddl("procedure", procedure_name);

select get_ddl("table", table_name);

```

## 1 控制流

### 1.1 使用 递归

```sql
with recursive t(n) as
(
    select 1
    union all
    select n+1 from t
)
select n from t limit 10
```

### 1.2 script block

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


### 1.3 使用while循环

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

### 1.4 使用 case

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

### 1.5 使用 loop

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

### 1.6 使用 for

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


### 1.7 使用 if

```sql
if (flag = 1) then
    return 'one';
elseif (flag = 2) then
    return 'two';
else
    return 'Unexpected input.';
end if;
```


## 2 特殊

### 2.1 使用变量和游标


Let 使用变量：
https://docs.snowflake.com/en/sql-reference/snowflake-scripting/let.html


使用游标：
https://docs.snowflake.com/en/developer-guide/snowflake-scripting/cursors.html




### 2.2 窗口相关 ⭐️

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


</br>

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


</br>

4. 使用lag

同一表中不同行

```sql
SELECT emp_id, year, revenue, 
       revenue - LAG(revenue, 1, 0) OVER (PARTITION BY emp_id ORDER BY year) AS diff_to_prev 
FROM sales 
ORDER BY emp_id, year;
```


</br>

5. 使用 QUALIFY ⭐️

参考：https://docs.snowflake.com/en/sql-reference/constructs/qualify

QUALIFY 对窗口函数的作用就像 HAVING 对聚合函数和 GROUP BY 子句的作用一样。

```sql
-- 筛选记录数 >=10 的所有用户的所有记录
select * from table_name
where ,,,
qualify count(1) over (partition by user_id) >= 10;

-- 取用户最新的一条记录
select * from t1
qualify row_number() over (partition by user_id order by created_at desc) = 1;
```

</br>

6. 使用 array_agg

参考：https://docs.snowflake.com/en/sql-reference/functions/array_agg

array_agg() 将一列输出转化为数组

```sql
SELECT ARRAY_AGG(O_ORDERKEY) WITHIN GROUP (ORDER BY O_ORDERKEY ASC)
FROM orders 
WHERE O_TOTALPRICE > 450000;
```




### 2.3 function & procedure


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



5. 使用包

```sql
create function add_one_to_inputs(x number(10, 0), y number(10, 0))
returns number(10, 0)
language python
runtime_version = 3.8
packages = ('pandas')
handler = 'add_one_to_inputs'
as $$
import pandas

def add_one_to_inputs(df):
  return df[0] + df[1] + 1

add_one_to_inputs._sf_vectorized_input = pandas.DataFrame
$$;
```

6. 使用 session 联通


https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-sprocs


```sql
create procedure procedure_name()
    returns Table()
    language python
    runtime_version = 3.8
    packages =('snowflake-snowpark-python')
    handler = 'main'
    as '# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col

def main(session: snowpark.Session): 
    # Your code goes here, inside the "main" handler.
    tableName = ''information_schema.packages''
    dataframe = session.table(tableName).filter(col("language") == ''python'')

    # Print a sample of the dataframe to standard output.
    dataframe.show()

    # Return value will appear in the Results tab.
    return dataframe';


call procedure_name();

drop procedure procedure_name();
```







### 2.4 动态构建 SQL 语句

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


### 2.5 其他

- 创建临时表
- 加载数据 https://docs.snowflake.com/en/user-guide/data-load-overview.html








## other


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

select '2'::int;
```

(5) 使用 exclude & json 取数据方法

```sql
select 
    *exclude(EXT_JSON_CONTENT),
    EXT_JSON_CONTENT:a as a,
    EXT_JSON_CONTENT:b as b
from t;
```

（6）使用 query_id

```sql
SELECT *
FROM TABLE(QUERY_HISTORY())
WHERE QUERY_ID = '查询ID';
```



-----------

> sql 中 group by 1 表示按第一列进行分组


<p class = "pyellow">生产环境还是要小心，多检查 pipeline 的状态</p>


坑， task create or replace 之后要 resume，

```sql
-- 检查 schedule， state 是 started 就行
SHOW TASKS LIKE 'task_name';

-- % 为 通配符
-- 匹配以 tmp 为子串的字符串
SHOW TASKS LIKE '%tmp%';
```

-----------

括号错位问题，造成的报错报到其它地方去了，报错是对的，少个括号真的不明显；

log 中类型问题，psql 使用 ::interger 进行类型转化，才不会报错；

sql 中打包进入 json 的都是字符串类型，需要格式转化。



-------------


参考资料：
- [Snowflake 官方文档](https://docs.snowflake.com/)
- [Book: SNOWFLAKE: THE DEFINITIVE GUIDE](https://www.snowflake.com/resource/snowflake-definitive-guide/)


