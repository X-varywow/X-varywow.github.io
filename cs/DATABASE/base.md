
## QUERY

- select
- ,,,




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



## DDL

`DDL`  (data definition language)


- create
- alter
- drop


```sql
drop table table_name;
```



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

