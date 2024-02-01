
SQL 指结构化查询语言，全称是 Structured Query Language。

SQL 让您可以访问和处理数据库，包括数据插入、查询、更新和删除。

CURD（create, update, read, delete）

------------


## select

</br>

### _base_

1. 选择固定列
```sql
SELECT column1, column2, ...
FROM table_name;
```

2. 选择全部列
```sql
SELECT * FROM table_name;
```

3. 选择唯一
```sql
SELECT DISTINCT column1, column2, ...
FROM table_name;
```

4. 限定条件
```sql
SELECT column1, column2, ...
FROM table_name
WHERE condition;
```

5. 对结果排序
```sql
SELECT column1, column2, ...
FROM table_name
ORDER BY column1, column2, ... ASC|DESC;
```

6. top&limit
```sql
SELECT * FROM Websites LIMIT 2;

select top 5 * from table
```

7. 正则选取
```sql
SELECT * FROM Websites
WHERE name REGEXP '^[^A-H]';
```
选取 name 不以 A 到 H 字母开头的网站：


</br>

### _join_



https://www.runoob.com/sql/sql-join.html


<img src='https://www.runoob.com/wp-content/uploads/2019/01/sql-join.png' style="zoom:60%">


</br> 

左连接，保留左边全部行，按左边行顺序与右边比较连接条件做连接，<u>一对多会产生多条连接记录</u>

利用笛卡尔积 （cross join）和中间表可以实现很多有用的逻辑，比如：每个列中附加全局统计信息(只有1行)


```sql
-- 以下几种写法等同

select * from a join b on a.id = b.id;

select * from a inner join b on a.id = b.id;

-- 隐式连接，不推荐使用，不太直观；join 为显性连接
select * from a,b where a.id = b.id
```


## insert


(1) 仅提供值

```sql
insert into table_name
values (value1,value2,value3,...);
```

(2) 指定列名及被插入的值：
```sql
insert into table_name (column1,column2,column3,...)
values (value1,value2,value3,...);
```

## update

```sql
update table_name
set column1 = value1, column2 = value2, ...
where condition;
```

## delete

```sql
DELETE FROM table_name
WHERE condition;
```

## other


| 运算符  | 描述                                                    |
| ------- | ------------------------------------------------------- |
| =       | 等于                                                    |
| <>      | 不等于。注释：在 SQL 的一些版本中，该操作符可被写成 ！= |
| >       | 大于                                                    |
| <       | 小于                                                    |
| >=      | 大于等于                                                |
| <=      | 小于等于                                                |
| BETWEEN | 在某个范围内                                            |
| LIKE    | 搜索某种模式                                            |
| IN      | 指定针对某个列的多个可能值                              |
| and     |                                                         |
| or      |                                                         |


> `<>` 和 `!=` 都用于比较两个值是否不相等， `<>` 是更标准标准的SQL操作符，使用更广泛。


| 通配符     | 描述                   |
| ---------- | ---------------------- |
| %          | 替代 0 个或多个字符    |
| _          | 替代一个字符           |
| [charlist] | 字符列中的任何单一字符 |


The COALESCE() function returns the first non-null value in a list.


---------------------

参考资料：
- [菜鸟教程](https://www.runoob.com/sql/sql-tutorial.html)
- [SQL语句汇总（三）——聚合函数、分组、子查询及组合查询](https://www.cnblogs.com/ghost-xyx/p/3811036.html)