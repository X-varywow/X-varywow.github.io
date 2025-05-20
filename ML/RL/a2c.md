
## A3C

Asynchronous Advantage Actor-Critic 由 deepmind 2016 年提出。

每个进程分别与环境进行交互学习；异步并行训练框架；

解决单个智能体与环境交互收集速度慢，训练难以收敛的问题；



|            | A3C                   | A2C                  |
| ---------- | --------------------- | -------------------- |
| 并行方式   | 异步（多线程/多进程） | 同步（统一批量更新） |
| 稳定性     | 比较不稳定            | 更稳定               |
| 实现复杂度 | 高                    | 较低                 |
| 并行效率   | 高，但容易冲突        | 略低但更新一致性更强 |


## A2C

A2C (Advantage Actor-Critic) 是 A3C 的同步版本，同步更新（多个环境并行采样，统一计算梯度）同样有效，且实现更简单






------------

参考资料：
- [参考](https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9tb2RlbGFydHMtbGFicy1iajQtdjIub2JzLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vY291cnNlL21vZGVsYXJ0cy9yZWluZm9yY2VtZW50X2xlYXJuaW5nL3BvbmdfQTNDL1BvbmctQTNDLmlweW5i)
- gpt