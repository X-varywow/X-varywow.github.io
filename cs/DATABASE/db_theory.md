## 一、数据库系统概述

`1.4`

**外模式**   -->   用户级
- 数据库用户能看到并允许使用的那部分局部数据的逻辑结构和特征的描述 

 ||   `外模式/模式映像`
 ||    保证了数据与程序间的逻辑一致性

**模式**     -->   概念级
- 数据库中全体数据的逻辑结构和特征的描述
- 一个数据库只有一个模式

 ||   `模式/内模式映像`
 ||    保证了数据的物理一致性

**内模式**   -->   物理级
- 它是对数据库存储结构的描述，是数据在数据库内部的表示方式 

>三级模式和二级映像的优点：
>- 保证数据独立性
>- 简化用户接口
>- 有利于数据共享
>- 有利于数据的安全保密

`1.3`

数据库系统（DBS）的组成：
- DB（是存储在计算机内、有组织的、可共享的数据和数据对象的集合）
- 用户
- 软件系统
- 硬件系统

`1.6`

数据库系统的**核心软件**：DBMS

数据库管理系统（DBMS）的主要功能：
- 数据定义
- 数据操纵
- 数据库运行管理
- 数据库的建立和维护
- 数据通信接口
- 数据存储、组织和管理

数据库管理系统（DBMS）的组成：
- 语言编译处理程序
- 系统运行控制程序
- 系统建立、维护程序
- 数据字典


`1.9`
- 层次模型
- 网状模型
- 关系模型
  - 一个关系实例对应一张由行和列组成的二维表
  - 每**一行元组**对应的列的属性值叫**分量**
  - 一个关系数据库文件中个条记录，前后顺序可颠倒
  - 关系模式的任何属性不可再分
- 面向对象模型

## 二、关系数据库

**候选码**，能唯一标识关系中元组的一个属性或属性集
- 唯一性
- 最小性

**主码**，被选用的候选码。

**超码**，能够决定所有属性，其中可以包含候选码外其他属性。

-------------

关系模型中的三类**完整性约束**：
- 实体完整性
- 参照完整性
- 用户自定义完整性

--------------

**关系代数运算符**：
- 1.集合运算符
  - `∪` `-` `∩` 和 广义笛卡尔积`x`
- 2.专门的关系运算符
  - 选取、投影、连接
- 3.算术比较运算符
- 4.逻辑运算符

_关系代数练习_

基于以下三个关系:  
- 学生S(SNO, SNAME, AGE)
- 课程C(CNO, CNAME, CT)
- 选修SC(SNO, CNO, SCORE)


其中SNO, SNAME, AGE, CNO, CNAME, CT, SCORE分别代表学号、学生姓名、年龄、课程编号、课程名称、课时、成绩。

用关系代数完成下列操作：

(1)	查询年龄小于17或者大于23的学生选修的课程编号。
`ΠCNO((σage<17 or age>23(S)) ⋈SC)`

(2)	给出重名的名字。
`ΠS.SNAME(σS.NAME=S1.NAME AND S.SNO!=S1.SNO (S☓ρS1(S)))`

(3)	查询“张三”同学不学的课程编号。
`ΠCNO(C) -ΠCNO(SC⋈σSNAME='张三' (S))`

(4)	给出既选修“高等数学”又选修“线性代数”的学生学号。
`ΠSNO(SC⋈(σCNAME='高等数学' (C)))∩ ΠSNO(SC⋈(σCNAME='线性代数' (C)))`

(5)	给出“数据库”课程成绩最高的学生学号。
`R1:= ΠSNO,SCORE(SC⋈(σCNAME='数据库' (C)))` (先将数据库成绩提出来，后面会简洁一些)
`(ΠSNO ( R1 - ΠR1.SNO, R1.SCORE(σR1.SCORE<R2.SCORE (R1☓ρR2(R1)))))`


## 三、SQL

一个数据库至少包含：
- 数据文件 `.mdf`
- 事务日志文件 `.ldf`

-----------------------

SQL功能：
- 数据**查询** `select`
- 数据**定义** `create` `drop` `alter`
- 数据**操纵** `insert` `update` `delete`
- 数据**控制**  `grant` `revoke`

---------------------


### 实验一：sql初步

```sql
-- 查询1986年出生的

select sno,sn,birth
from s
where birth like'1986%'

-- 或者 where year(birth)=1986
```

```sql
--选取3次作业总分前三

select top3 sno,cno,hw1+hw2+hw3 as total
from homework
order by total desc
```

### 实验二：sql子查询

```sql
--查询与xxx同一班级的其他学生信息

--连接查询
select s.*
from s, s as s1
where s.class=s1.class and s1.sn=xxx and s.sn!=xxx

--子查询
select s.*
from s
where class=(select class from s where sn=xxx) and s.sn!=xxx
```

```sql
-- 查询K001和M001都没有选修的学生信息

select sno,cno,hw1,hw2,hw3
from homework
where sno not in (
    select sno
    from homework
    where cno='K001' or cno='M001'
)
```

```sql
-- 查询学时最少的课程

select cn
from c
where ct <=all(select ct from c)

--或

select cn
from c
where not exists(select * from c as c1 where c.ct>c1.ct)
```

### 实验三：sql聚合函数

```sql
-- 查询学生人数

select count(*)
from homework
where cno='K001'
```

```sql
-- 查询多少个不同的班级

select count(distinct class)
from s
```

```sql
-- 查询作业平均分

select avg(hw1)
from c join homework on c.cno=homework.cno and cn='python'
```

```sql
-- 查询选课人数最多的两名课程，给出课程号

select top 2 cno
from homework
group by cno
order by count(sno) desc
```

```sql
-- 查询两个以上男生选修的课程编号。（包含两个）
-- 因为一个SN对应多个SNo时出错
select CNO
from s join homework on s.sno=homework.sno and sex='男'
group by CNO
having count(*)>=2
```

```sql
-- 9. 查询每个同学的选课门数，如果没有选修则选课门数为0。

select s.sno,sn,count(cno) as 选课数
from s left join homework on s.sno=homework.sno
group by s.sno,sn
order by 选课数 desc

-- 方法二：

select s.sno,sn,count(cno) as 选课数
from s join homework on s.sno=homework.sno
group by s.sno,sn

union

select sno,sn, 0 as 选课数
from s
where sno not in
(select sno from homework)
```

### 实验四：数据更新与约束⭐

```sql
-- 创表，约束，主外键
CREATE TABLE Book(
	BNo CHAR(15) PRIMARY KEY,
	BName	nVARCHAR(50),
	Publish	nVARCHAR(50),
	PDate	datetime,
	BAuth	nVARCHAR(30),
	bprice	NUMERIC(4,1),
	binprice	NUMERIC(4,1),
	BCount	INT check(Bcount>=0),
	check(binprice<bprice)
);

CREATE TABLE BookSell(
	SDate	datetime,
	BNO	CHAR(15) foreign key references Book(bno),
	SCount	int,
	SMoney	smallmoney
);

--新进图书
insert into Book values ('9787115457004','数据库原理及应用教程','人民邮电出版社','2017-11-1','陈志泊',49.5,35,200);

--卖书退书
insert into booksell values('2020-2-1','9787115457004',20,800);
update book  set bcount-=20 where bno='9787115457004';
insert into booksell values('2020-2-2','9787115457004',3,132);
update book  set bcount-=3 where bno='9787115457004';
insert into booksell values('2020-2-3','9787115457004',-1,(select smoney/scount*-1 from booksell where scount=3));
update book  set bcount+=1 where bno='9787115457004';

-- 补充：delete from<表名> [where<条件>]

-- 不加where可以修改多行
```

## 四、关系数据库理论

关系模式中各属性之间相互依赖，相互制约的联系称为数据依赖。数据依赖一般分为 **函数依赖** 和 **多值依赖**。

**函数依赖** 是关系模式中属性之间的一种逻辑依赖关系。

- 完全函数依赖：属性集 `X` 的任何真子集都推不出 `Y`。
- 部分函数依赖：属性集 `X` 的某个真子集推出了 `Y`。

>**求解最小函数依赖集**：
>1. 将每个函数依赖的右边变成单属性
>2. 去掉每个函数依赖左边的冗余属性
>3. 去掉冗余的函数依赖

> `F+` = `G+` 时, 函数依赖集 `F` 和 `G` 等价

------------------------

**闭包**：`F+` 是被 `F` 逻辑蕴涵的全部函数依赖集合（字面意思）。

> 若 `X+`包含了 `R` 的全部属性，则属性集 `X` 是 `R` 的一个码。


-------------------------

**1NF** ：关系中每个属性都是不可再分的原子项

&emsp;&emsp;`1NF` -> `2NF` 【消除了非主属性对主码 **部分函数依赖**】

**2NF** ：不存在非主属性对主码部分函数依赖（每个非主属性都完全函数依赖于主码）

&emsp;&emsp;`2NF` -> `3NF` 【消除非主属性对主码的 **传递函数依赖**】

**3NF** ：每个非主属性都不传递函数依赖于主码.

**BCNF** ：如果关系模式R∈1NF，且对于R的每个函数依赖X→Y（Y不属于X），决定因数X都包含了R的一个候选码，那么R∈BCNF。


------------------------

**书上例题**：
- 求函数依赖集的闭包 `4-2`
- 求解候选码 `4-3` `4-4` `4-5`
- 求最小函数依赖集 `4-6` `4-7` `4-8` `4-9`
- 关系模式的分解 `4-10~14`
- 范式 `4-15~21`


**判定问题：**
- `无损连接性判定`
  ①画表格（行是单个属性，列是分解的属性集）
  ②属性存在于属性集中的、能函数依赖推导出来的
  ③若有一行都满足②，则是无损连接分解
  ④特殊判定，R1 ∩ R2 -> （R1-R2 或 R2-R1)
- `函数依赖保持性判定`
  看看每个分解的部分得出的依赖，能不能凑齐总的函数依赖

**其他问题**：
- 任何一个包含两个属性的关系模式一定满足 BCNF
- { AB->CD , A->D } 最高属于 1NF
  



## 五、数据库安全保护

##### 一、安全性控制
  - `目的`：【防止非法使用造成数据的泄露、更改和破环】
  - `方法`：
	- 用户标识和鉴定
	- 用户存取权限控制
	- 定义视图
	- 数据加密和审计
  - 当用户要访问数据库时，必须要有 `登录账号` `用户账号`

---------------------------

##### 二、完整性控制
`目的`：【防止合法用户加入不符合语义的数据】
`方法`：
- 约束
- 默认值
- 规则
- 存储过程
- 触发器

--------------------------

##### 三、并发性控制
`目的`：【防止多个同时存取同一数据造成不一致】
`方法`：封锁（普遍采用），时标

--------------------------

##### 四、数据恢复

`三种模式`
- 完整恢复模式
- 大容量日志记录恢复模式（使用数据库备份和日志备份来还原数据库）
- 简单恢复模式（可执行完全数据库备份和增量数据库备份来还原数据库）

`基本原理`：利用数据的冗余（登记日志文件，数据转储）

### 实验五：数据库安全管理

```sql
-- 授权权限
use test
go
grant insert on dbo.t1 to zhang with grant option
go

--收回权限(级连)
revoke insert on bdo.t1 from zhang cascade
```


## 六、数据库设计

①**需求分析**
- 数据流图（表达了数据和处理过程的关系）
- 数据字典（对系统中数据的详细描述）
  
②**概念结构设计**
- 概念模型（ER图）
  - ER图冲突：属性冲突、命名冲突、结构冲突
  

③**逻辑结构设计**
1. 初始关系模式设计
2. 关系模式规范化
3. 模式的评价和改进

④**物理结构设计**
- 确定表、字段、索引

⑤**实施**
- 建立实际数据库结构

⑥**运行和维护**

## 七、高级应用


**事务**：数据库系统中执行的一个工作单位，DBMS并发控制的基本单位。
- 原子性 `要么不做，要么全做`
- 一致性
- 隔离性
- 持久性

--------------------

**批处理**：一个批处理语句在一起通过解析才执行，每个批处理（两个 `go` 之间）单独执行。

--------------------

### 实验六：Transaction-sql

`事务练习`
```sql
--zhang转账给li
use ***
begin transaction
	declare @num=balance from account where name='zhang'
	if(@ba>100)
		begin
		update account set balance+=100 where name='li'
		update account set balance-=100 where name='wang'
		commit
		end
	else
		rollback
```

`case练习`
```sql
select name,balance,
	case
		when balance>1000 then 'high'
		when balance<1000 and balance>0 the 'low'
		else 'other'
	end as 分类
from account
```

`时间函数练习`
```sql
-- 明年十月一星期几？
select datename(weekday,cast(year(getdate())+1 as varchar)+'-10-01')
```

`保留两位小数`
```sql
select sno,convert(decimal(10,2),avg(score))
from sc
group by sno
```

`自定义函数`
```sql
--查询成绩在XY之间的信息
create function query(@x int,@y int)
returns table
as return(
select * from sc
where score>=@x and score<=@y>)

--执行
declare @x int,@y int
set @x=60
set @y=80
select * from query(@x,@y)
```

### 实验七：存储过程&触发器

```sql
-- 存储过程，返回及格率
create procedure query
@vcno char(10),
@vi float output
as
declare @a float,@b float
select @a=count(*) from sc where cno=@vcno and grade>=60
select @b=count(*) from sc where cno=@vcno
set @vi=@a/@b
print @a
print @b
print @vi

declare @ans float
exec query 'cs', @ans output
select cast(@ans*100 as varchar)+'%' as '及格率'

-- 触发器，限定删除
create trigger del_s on dbo.s
after delete
as
declare @s char(10)
select @s =sno from deleted
if @s in (select sno from sc)
begin
	print '该生已有成绩，不能删除'
	print @s
	rollback
end

```


### 实验八：备份&还原

```sql
--对数据库进行完全备份
backup database tmp to disk='c:\tmp\tmpfull.bak'

--对数据库进行差异备份
backup database tmp to disk='c:\tmp\tmpdiff.bak' with differential

--备份日志
--日志记录着对数据库的更新操作
backup log tmo to disk='c:\tmp\tmplog.bak'

--完整恢复模式，需要备份和还原事务日志
alter database abc set recovery full

restore database tmp from disk='c:\blog\tmpfull1.bak' with norecovery
restore log tmp from disk='c:\blog\tmplog1.bak' with norecovery
restore log tmp from disk='c:\blog\tmplog2.bak' with stopat='2020-12-08 22:29:27',recovery
```

-------------------

**遗漏了**：视图，索引

**推荐阅读**：
- [数据库事务的四种隔离级别](https://www.cnblogs.com/dingpeng9055/p/11190203.html)