
题目来源：leetcode

## Day1 选择

### [595. 大的国家](https://leetcode.cn/problems/big-countries/)

```sql
select name, population, area
from World
where area>=3000000 or population>=25000000
```

### [1757. 可回收且低脂的产品](https://leetcode.cn/problems/recyclable-and-low-fat-products)

```sql
select product_id
from Products
where low_fats="Y" and recyclable="Y"
```

### [584. 寻找用户推荐人](https://leetcode.cn/problems/find-customer-referee/)

```sql
select name
from customer
where referee_id != 2 or referee_id is null
```
null 无法与确定的值比较

### [183. 从不订购的客户](https://leetcode.cn/problems/customers-who-never-order/)

```sql
select Name as Customers from Customers
where Id not in (
    select CustomerId from Orders
);
```

子查询

## Day2 排序&修改


### [1873. 计算特殊奖金](https://leetcode.cn/problems/calculate-special-bonus/)

```sql
select employee_id, if(
    employee_id % 2 = 1 and name not like 'M%', salary, 0
) as bonus
from Employees
order by employee_id
```

like 与通配符一起使用，% 替代 0 个或多个字符

### [627. 变更性别](https://leetcode.cn/problems/swap-salary/)

```sql
update salary set sex = if( sex='m','f','m')
```

if 语句

字段 = if(条件，结果1，结果2)


### [196. 删除重复的电子邮箱](https://leetcode.cn/problems/delete-duplicate-emails/description/)

```sql
delete p1
from Person p1, Person p2
where p1.email = p2.email and p1.id>p2.id
```

采用自连接的方式，将一个表重命名；

从 p1 表中删除满足 where 条件的记录


## Day3 字符串处理函数/正则


### [1667. 修复表中的名字](https://leetcode.cn/problems/fix-names-in-a-table/)

```sql
select user_id,
concat(upper(left(name,1)),lower(right(name, length(name)-1))) as name
from users
order by user_id
```
concat(), upper(), left(), right()

### [1484. 按日期分组销售产品](https://leetcode.cn/problems/group-sold-products-by-the-date/)

```sql
select sell_date, 
count(distinct product) as num_sold, 
group_concat(distinct product order by product separator ',') as products
from Activities
group by sell_date
order by sell_date
```

group by 配合聚合函数使用：
- count()
- sum()
- avg()
- max()
- min()

group_concat(group separator ',')


### [1527. 患某种疾病的患者](https://leetcode.cn/problems/patients-with-a-condition/)

```sql
select *
from patients
where conditions like 'DIAB1%' or conditions like '% DIAB1%'
```

`*` 代表全部

`%` 替代 0 个或多个字符

## Day4 组合查询 & 指定选取

### [1965. 丢失信息的雇员](https://leetcode.cn/problems/employees-with-missing-information)

`union all` 取并集，相较于 `union` 不进行去重操作，效率较高；

`group by` 与 `having count(*) = 1` 作为筛子；

```sql
select employee_id 
from(
    select employee_id,name from Employees
    union all
    select employee_id,salary from Salaries
) t
group by employee_id
having count(*) = 1
order by employee_id;
```

### [1795. 每个产品在不同商店的价格](https://leetcode.cn/problems/rearrange-products-table)

学到了："store1" as store 这种写法

```sql
select product_id, "store1" as store, store1 as price from Products where store1 is not null
union all
select product_id, "store2" as store, store2 as price from Products where store2 is not null
union all
select product_id, "store3" as store, store3 as price from Products where store3 is not null;
```

### [608. 树节点](https://leetcode.cn/problems/tree-node/)

```sql
select id,
(
    case
    when p_id is null then "Root"
    when id in (select p_id from tree) then "Inner"
    else "Leaf"
    end
) as Type
from tree;
```

### [176. 第二高的薪水](https://leetcode.cn/problems/second-highest-salary)

- row_number() 不会跳过 rank，重复时为不同 rank
- dense_rank() 不会跳过 rank，重复时为同 rank
- rank() 会跳过 rank，重复时为同 rank

方法一：
```sql
select (
    select salary
    from (
        select distinct salary,
        dense_rank() over(order by salary desc) rk
        from employee

    ) t
    where rk = 2

) as  secondhighestsalary
```

方法二：
```sql
select max(salary) as secondhighestsalary
from employee
where salary<(select max(salary) from employee);
```


## Day5 合并

### [175. 组合两个表](https://leetcode.cn/problems/combine-two-tables)

```sql
select firstname, lastname, city, state
from person a
left join address b
on a.personid = b.personid;
```

left join 属于 outer join ，附表中可能出现 null 值

### [1581. 进店却未进行过交易的顾客](https://leetcode.cn/problems/customer-who-visited-but-did-not-make-any-transactions)

```sql
select customer_id, count(visit_id) as count_no_trans
from visits
where visit_id not in (select visit_id from transactions)
group by customer_id;
```

### [1148. 文章浏览 I](https://leetcode.cn/problems/article-views-i/)

```sql
select distinct author_id as id
from views
where author_id = viewer_id
order by id;
```

## Day6 合并

### [197. 上升的温度](https://leetcode.cn/problems/rising-temperature)

```sql
select a.id
from weather a, weather b
where a.recorddate = date_add(b.recorddate, interval 1 day) and a.temperature > b.temperature;
```

### [607. 销售员](https://leetcode.cn/problems/sales-person)

```sql
select name
from salesperson
where sales_id not in(
    select sales_id
    from orders
    where com_id = (select com_id from company where name="RED")
);
```

## Day7 计算函数

### [1141. 查询近30天活跃用户数](https://leetcode.cn/problems/user-activity-for-the-past-30-days-i/)

```sql
select activity_date as day,count(distinct user_id) as active_users
from activity
group by activity_date
having datediff('2019-07-27', activity_date) < 30 and datediff('2019-07-27', activity_date) > 0;
```

### [1693. 每天的领导和合伙人](https://leetcode.cn/problems/daily-leads-and-partners/)

```sql
select date_id,make_name,
count(distinct lead_id) as unique_leads,
count(distinct partner_id) as unique_partners
from dailysales
group by date_id, make_name;
```

group by 即为分组汇总，要求 select 出的结果字段都是可汇总的

### [1729. 求关注者的数量](https://leetcode.cn/problems/find-followers-count/)

```sql
select user_id,count(distinct follower_id) as followers_count
from followers
group by user_id
order by user_id;
```

## Day8 计算函数

### [586. 订单最多的客户](https://leetcode.cn/problems/customer-placing-the-largest-number-of-orders/)

写法一：
```sql
select customer_number
from 
(
    select customer_number, count(*) as cnt
    from orders
    group by customer_number
    order by cnt desc
) t
limit 1;
```

写法二：
```sql
select customer_number
from orders
group by customer_number
order by count(*) DESC
limit 1;
```


### [511. 游戏玩法分析 I](https://leetcode.cn/problems/game-play-analysis-i)

```sql
select player_id, min(event_date) as first_login
from activity
group by player_id;
```

### [1890. 2020年最后一次登录](https://leetcode.cn/problems/the-latest-login-in-2020/)

```sql
select user_id, max(time_stamp) as last_stamp
from logins
where year(time_stamp) = 2020
group by user_id;
```

### [1741. 查找每个员工花费的总时间](https://leetcode.cn/problems/find-total-time-spent-by-each-employee)

```sql
select event_day as day,
emp_id,
sum(out_time) - sum(in_time) as total_time
from employees
group by emp_id, event_day;
```

## Day9 控制流

### [1393. 股票的资本损益](https://leetcode.cn/problems/capital-gainloss/)

```sql
select stock_name,
sum(if(operation='Buy', -price, price)) as capital_gain_loss
from stocks
group by stock_name;
```

### [1407. 排名靠前的旅行者](https://leetcode.cn/problems/top-travellers/)

```sql
select name,
ifnull(d,0) as travelled_distance
from(
    select user_id, sum(distance) as d
    from rides
    group by user_id
) a
right join users on users.id = a.user_id
order by travelled_distance desc, name asc
;
```

order by 设置多个规则，这里使用right join，因为右边的表是 id 较多的；


### [1158. 市场分析 I](https://leetcode.cn/problems/market-analysis-i/)

```sql
select user_id as buyer_id,
join_date,
ifnull(cnt, 0) as orders_in_2019
from users
left join (
    select buyer_id,count(buyer_id) as cnt
    from orders
    where year(order_date) = 2019
    group by(buyer_id)
) t
on users.user_id = t.buyer_id;
```

## Day10 过滤

### [182. 查找重复的电子邮箱](https://leetcode.cn/problems/duplicate-emails/)

```sql
select email
from person
group by email
having count(*) > 1;
```

### [1050. 合作过至少三次的演员和导演](https://leetcode.cn/problems/actors-and-directors-who-cooperated-at-least-three-times)

```sql
select actor_id, director_id
from actordirector
group by actor_id, director_id
having count(*) > 2;
```

### [1587. 银行账户概要 II](https://leetcode.cn/problems/bank-account-summary-ii)

```sql
select name, n as balance
from (
    select account,sum(amount) as n
    from transactions
    group by account
    having sum(amount) > 10000
)t left join users
on t.account = users.account;
```

### [1084. 销售分析III](https://leetcode.cn/problems/sales-analysis-iii/)

```sql
select product_id, product_name
from sales join product
using(product_id)
group by product_id
having sum(sale_date between '2019-01-01' and '2019-03-31') = count(sale_date);
```

使用 using(id) 简化 on a.id = b.id