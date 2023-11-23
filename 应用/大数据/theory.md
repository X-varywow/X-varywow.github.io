
## preface

关键技术: 分布式存储, 分布式处理



- 批计算处理 mapreduce, spark
- 流计算
- 图计算
- 查询分析计算

## (2) 架构设计

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20230330234704.png">

### 2.1 数据采集层

日志采集；分流、限流、削峰、压测

数据同步；
- 离线
- 实时；（秒级数据刷新，如双11数据大屏）
  - 解析 mysql 的 binlog 日志来实时获得增量的数据更新，并通过消息订阅模式来实现数据的实时同步
- 分库分表、增量+全量
- 冗余数据来处理数据漂移


### 2.2 数据计算层

数仓的分层：
- ODS，operational data store
- DWD，data warehouse detail
- DWS，data warehouse summary
- ADS，application data store


[ODS& DWD& DWS& ADS 数仓分层](https://blog.csdn.net/young_0609/article/details/103088253)

### 2.3 数据服务层

### 2.4 数据应用层





-----------

OLAP & OLTP

联机分析处理 (OLAP) 系统的主要用途是分析聚合数据，而联机事务处理 (OLTP) 系统的主要用途是处理数据库事务

使用 OLAP 来分析 OLTP 收集的数据，OLAP 针对复杂的数据分析和报告进行了优化，OLTP 则针对事务处理和实时更新进行了优化


------------------

参考资料：
- 《大数据之路；阿里巴巴大数据实践》
- [阿里云-大数据学习路线](https://developer.aliyun.com/learning/roadmap/bigdata)
- [黑马程序员大数据学习路线图](https://yun.itheima.com/subject/cloudmap/index.html)