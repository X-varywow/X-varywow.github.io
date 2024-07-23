
欢迎来到 数据库 ；

大数据相关，[请参考](应用/大数据/)



</br>

_基础术语_

主键，指的是一个列或多列的组合，其值能唯一地标识表中的每一行，通过它可强制表的实体完整性。主键主要是用与其他表的外键关联，以及文本记录的修改与删除。

外键：是一个表中的字段，与另一个表的主键相关联。（外键用于在两个表之间建立链接，确保引用的数据的完整性）

外键约束：外键字段的值必须是另一个表中的有效主键或唯一键。（数据库规则，外键引用的数据必须在另一张表存在，外键与另一张表的主键的关联关系不定）（确保了两个表之间的数据关联性和引用完整性）

索引：使用索引可快速访问数据库表中的特定信息。索引是对数据库表中一列或多列的值进行排序的一种结构。类似于书籍的目录。

归档：将数据从在线数据库（正在使用中）转移到一个长期存储的数据库或文件系统。（减少在线数据库的负载，提高性能释放空间）


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



> 表 & 视图

[Difference Between View and Table](https://www.geeksforgeeks.org/difference-between-view-and-table/)

视图图是 SQL 查询的结果，它是一个虚拟表。

<u>视图更像是存储的查询，所以执行速度比较慢。</u>






</br></br>

_性能优化_


[SQL 性能优化1](https://www.cnblogs.com/youzhibing/p/11909821.html)

```sql
-- 使用 exists 替代 dsitinct
select distinct t1.* from t2 left join tb t1 on t1.id = t2.id;

select * from t1 where exists(select id from t2 where t1.id = t2.id);


-- 减少临时表
```



