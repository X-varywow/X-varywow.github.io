
## QUERY


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20231022224708.png"  style="zoom:50%;">



## DML

`DML`  (data manipulation language)

- insert
- merge
- update
- delete
- truncate

```sql
INSERT INTO employees
  VALUES
  ('Lysandra','Reeves','1-212-759-3751','New York',10018),
  ('Michael','Arnett','1-650-230-8467','San Francisco',94116);

-- 覆盖写
INSERT overwrite INTO employees_copy
select *
from employees;
```

```sql
-- 删除 t1 所有数据
delete from t1;


-- truncate 通常会比 delete 更快，因为不会扫描整个表
-- restart identity 重置该表的序列计数器（如果它包含任何自增主键）
truncate table t1 restart identity;
```

仍会保留结构定义和表本身




## DDL

`DDL`  (data definition language)


- create
- alter
- drop


```sql
-- 使用 a 表结构创建 b 表
create table b like a;

drop table table_name;
```

> drop 命令用于<u>删除整个表及其结构</u>，与 DML 中的 delete truncate 不同。



## DCL

`DCL`  (data control language)


- grant
- revoke


```sql
GRANT ALL PRIVILEGES ON table_name TO role role_name;
```


------------

参考资料：
- https://docs.snowflake.com/en/sql-reference-commands

