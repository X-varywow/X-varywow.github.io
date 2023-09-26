

使用 rancher（kubernetes management）

uvicorn --host "0.0.0.0" --port 8888 "dataai.main:app"


> Ops (Operations), 运维

## dataai-eks-prod

根据 GET 得：访问 IP `18.209.144.235:443`，部署在美国的服务器， 与 sagemaker 同一片区域

端口 443 ？？？

端口 8888 用于运维检查 pod 是否正常，不能清掉


服务部署在 /app 中

启动方式
```bash
/bin/bash/ -c 'sh -x scripts/run.sh start'
```

uvicorn --host "0.0.0.0" --port 8888 "dataai.main:app"


弄不了，pod mem 限制了 1gb; 顶多变成一个转发的地方

------------

参考资料：
- [为 Pod 和容器管理资源](https://kubernetes.io/zh-cn/docs/concepts/configuration/manage-resources-containers/)
- [运维的未来是平台工程](https://www.ruanyifeng.com/blog/2023/03/platform-engineering.html)