
欢迎来到 数据库 ；

大数据相关，[请参考](应用/大数据/)

</br>

_基本语言_

- `DML`  (data manipulation language)
  - select
  - update
  - insert
  - delete
- `DDL`  (data definition language)
  - create
  - alter
  - drop
  - truncate
- `DCL`  (data control language)
  - grant
  - revoke


</br>

_基础术语_

主键，指的是一个列或多列的组合，其值能唯一地标识表中的每一行，通过它可强制表的实体完整性。主键主要是用与其他表的外键关联，以及文本记录的修改与删除。

外键：外键用于关联两个表。

外键约束：

索引：使用索引可快速访问数据库表中的特定信息。索引是对数据库表中一列或多列的值进行排序的一种结构。类似于书籍的目录。


</br>

_常见问题_

>单引号 和 双引号；

单引号用于字符串值，不用于列名表名
```sql
select *
from students
where name = 'john'
```

双引号用的较少，用于区分大小写的标识符，用于带有空格或特殊字符的标识符
```sql
select "Name"
from students;

select "First Name"
from students;
```

> `COUNT(1)` 与 `COUNT(*)`

在SQL中，`COUNT(1)`是一个聚合函数，用于计算指定列或表中非空行的数量。具体来说，`COUNT(1)`会将每一行都视为非空，然后统计这些非空行的数量。

`COUNT(1)`与`COUNT(*)`是等价的，它们的作用都是计算行数。使用`COUNT(1)`通常被认为是一种最佳实践，因为它更加简洁且效率更高。




```sql
delete from table_name where ...;

insert into table_name ()
values 
(col1, col2, col3...),
(),
...


```



