
[milvus 官网](https://milvus.io/)


虚拟机中， ctrl + shift + v 才是粘贴


## 简介

Milvus 是一款云原生向量数据库, 共享存储架构，存储计算完全分离，计算节点支持横向扩展。


基本概念
- collection 相当于 table
- entity 由一组表示真实世界对象的字段 fields 组成（实际的数据内容）
- field 结构化数据或向量(列名)



## pymilvus

milvus 使用了 19530 和 9091 两个端口
- 19530 用于 gRPC
- 9091 用于 RESTful API


[参考代码1](https://github.com/JackLCL/search-video-demo/blob/master/search/controller/indexer.py)

[官方文档](https://milvus.io/docs/example_code.md#Run-Milvus-using-Python)




## 部署

### 方式1：走虚拟机 ubuntu

下载个 vm 虚拟机：http://www.winwin7.com/soft/17946.html

去镜像找个系统：http://iso.mirrors.ustc.edu.cn/ubuntu-releases/

装好后，更换下载源：https://zhuanlan.zhihu.com/p/61228593

安装 docker，还挺复杂: https://zhuanlan.zhihu.com/p/143156163

按照原步骤，加个 sudo make all ， 等待 编译

docker-compose 对五个容器进行管理


更改docker 源，更换镜像快多了

```bash
sudo vi /etc/docker/daemon.json

{
  "registry-mirrors": [
    "https://dockerproxy.com",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com",
    "https://ccr.ccs.tencentyun.com"
  ]
}

sudo service docker restart
```

安装 yarn: https://developer.aliyun.com/article/762675

切换源：


```bash
# 获取以图搜视频代码
git clone -b 0.7.1 https://github.com/JackLCL/search-video-demo.git

# 构建前端界面 docker 和 api docker 镜像
cd search-video-demo

make all
```
> 没走通，很麻烦，nodejs 报错


### 方式2：sagemaker

```bash
# 安装 docker-compose
wget https://github.com/docker/compose/releases/download/v2.19.1/docker-compose-linux-x86_64
chmod +x docker-compose-linux-x86_64
sudo cp docker-compose-linux-x86_64 /usr/bin/docker-compose

# 获取 docker 配置 并启动 docker
wget https://github.com/milvus-io/milvus/releases/download/v2.2.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
sudo docker-compose up -d


# 检查状态
sudo docker-compose ps
```

| SERVICE    | PORTS                                       |
| ---------- | ------------------------------------------- |
| etcd       | 2379-2380/tcp                               |
| minio      | 9000/tcp                                    |
| standalone | 0.0.0.0:9091->9091/tcp, :::19530->19530/tcp |





---------------

参考资料：
- http://www.yishuifengxiao.com/2022/12/27/%E5%90%91%E9%87%8F%E6%90%9C%E7%B4%A2%E6%95%B0%E6%8D%AE%E5%BA%93milvus%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B/
- [Milvus实战｜ 以图搜视频系统](https://zhuanlan.zhihu.com/p/139847892)
- https://aistudio.baidu.com/aistudio/projectdetail/1910614
- https://milvus.io/docs/install_standalone-docker.md





