

## _k8s_

Kubernetes (K8s) 是 Google 基于 Borg 开源的容器编排调度引擎，是云原生应用的基石，已成为事实标准。基于规范描述集群架构，定义服务的最终状态，并使系统自动地达到和维持该状态。


Kubernetes 提供了一个可弹性运行分布式系统的框架。便于 **管理容器**，确保容器发生故障时可由另一个容器保障业务安全等。

- 服务发现和负载均衡
- 存储编排
- 自动部署和回滚
- 自我修复
- 密匙与配置管理

常见的三个容器编排器：Docker Compose, Swarm, Kubernetes


<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20240108213620.png" style="zoom:70%">

------------

```bash
# 若找不到需要 -n <namespace>
kubectl get pods

# 查看 last state 中 reason 和 exit code 可以看到服务重启原因
kubectl describe pod pod-name -n prod
```

- kubectl 命令行工具，用于与 k8s 集群交互
- -n prod 指定命名空间为 prod
- describe 显示特定资源详细新信息
- pod 容器组，是k8s 集群中可以运行容器的最小单位



------------

last state: terminated
reason: error
exit code： 137


137 对应 SIGKILL(kill -9) 信号，说明容器被强制重启
可能原因：
- 检活失败
- 被宿主机杀掉（k8s 的 QoS）


```bash
kubectl get nodes

# 找到对应重启服务的节点
kubectl describe node <node-ip>

kubectl top node <node-ip>
```

参考：[记录一次 K8s pod被杀的排查过程](https://www.cnblogs.com/xtf2009/p/17947545)


## rancher

Rancher 是一个开源的容器管理平台，使用户能够轻松地部署和管理容器集群，实现高效的容器化应用部署和运维。

keyword: 问题排查, eks 

Cluster -> nodes 输入 10-210-999-999 ip 可找到对应的机器；或 pod 搜

kubectl shell 在右上角