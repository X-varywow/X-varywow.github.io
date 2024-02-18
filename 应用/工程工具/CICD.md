

Continuous Integration, Continuous Delivery

CI/CD 是一种通过在应用开发阶段引入自动化来频繁向客户交付应用的方法。

CI/CD 的核心概念是持续集成、持续交付和持续部署。

CI/CD 可让持续自动化和持续监控贯穿于应用的整个生命周期（从集成和测试阶段，到交付和部署）。

**总结：**应用开发中高度持续自动化和持续监控的 pipeline


## jenkins

Jenkins是一个开源的自动化构建工具，用于持续集成和持续交付。它提供了一个可视化的界面，可以帮助开发人员自动化构建、测试和部署软件项目。Jenkins支持各种编程语言和工具，并可以与许多其他开发和部署工具集成，如Git、Maven、Docker等。

https://www.jenkins.io/

[jenkins + python 教程](https://www.jenkins.io/zh/doc/tutorials/build-a-python-app-with-pyinstaller/)

https://www.jenkins.io/zh/doc/pipeline/tour/getting-started/

[回滚操作](https://segmentfault.com/a/1190000039164950)，回滚对于线上服务是一个必要的选项，能最大减少重新构建的时间。不然人为改动代码，再检查发布又消耗一段时间


静态扫描，通常是指自动化构建或部署过程中加入静态代码分析，以发现潜在的错误、安全漏洞、质量问题等。这一重要的质量保证措施，在 持续集成CI 中尤为常见。


## Maven

Maven是一个 Java 的项目管理和构建工具。

Jenkins 更加灵活，适用各种项目。



## Github Actions

https://yeasy.gitbook.io/docker_practice/ci/actions


## 灰度测试

灰度测试是在软件开发的后期阶段，对 **软件的一部分真实用户** 进行测试，以验证其功能和稳定性；

而内测是在软件开发的初期阶段，由软件开发团队内部的成员进行测试，以发现和修复软件的问题。


## rancher

Rancher 是一个开源的容器管理平台，使用户能够轻松地部署和管理容器集群，实现高效的容器化应用部署和运维。


## 监控

建立监控，保证可用性、正确性

代码上设置监控报警点

Grafana 监控看板；Prometheus 是一种开源监控和警报工具套件；python 使用参考：[fastapi](Python/web/fastapi)

```python
from prometheus_fastapi_instrumentator import Instrumentator
```

------------


一些术语：

**Mock 数据** 是指在软件开发，尤其是前端开发和测试环节中，用来模拟真实数据的测试数据。

Mock 数据可以用来模拟后端API的返回结果，让开发者在真实的后端服务尚未完成或不可用的时期内，能够独立于后端进行前端功能的开发与调试。

-------------

参考资料：
- chatgpt