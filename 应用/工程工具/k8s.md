

Kubernetes (K8s) 是 Google 基于 Borg 开源的容器编排调度引擎，是云原生应用的基石，已成为事实标准。基于规范描述集群架构，定义服务的最终状态，并使系统自动地达到和维持该状态。


Kubernetes 提供了一个可弹性运行分布式系统的框架。便于 **管理容器**，确保容器发生故障时可由另一个容器保障业务安全等。

- 服务发现和负载均衡
- 存储编排
- 自动部署和回滚
- 自我修复
- 密匙与配置管理

常见的三个容器编排器：Docker Compose, Swarm, Kubernetes


------------

```bash
kubectl get pods

kubectl -n prod describe pod pod-name
```

- kubectl 命令行工具，用于与 k8s 集群交互
- -n prod 指定命名空间为 prod
- describe 显示特定资源详细新信息
- pod 容器组，是k8s 集群中可以运行容器的最小单位