

容器，用于实现环境打包隔离。

Docker 是一个用 Go语言实现的开源项目，对 Linux 容器的一种封装（虚拟化），提供简单易用的容器使用接口。



## 基础

- image （镜像，一个只读的模板，包含应用程序及其依赖，用来创建容器）
- container (容器实例， 独立的应用运行环境) (**拉取 image 创建 container 实例**)
- dockerfile（配置文件）


```bash
# 将 dockerfile 交给 docker 编译，之后会创建可执行程序 image
docker build

# 将 image 加载到内存，作为一个 container 运行起来
docker run

# 列出所有运行过的容器，包括正在运行和已经停止的容器。
docker ps -a
```

</br>

### 1.1 demo: hello

```bash
docker image pull library/hello-world

docker image ls

docker container run hello-world
```

> run 会比 runc 多出自动拉取、创建容器的步骤


</br>



### 1.2 run

```bash
# 最基本运行（前台运行）
docker run <镜像名>

# 后台运行容器，并指定名称
docker run -d --name my_container <镜像名>

# 运行并进入容器的交互式终端（退出终端容器会停止）
docker run -it --name my_container <镜像名> /bin/bash

# 运行容器并映射端口（主机端口:容器端口）
docker run -d -p 8080:80 --name my_web nginx

# 运行容器并挂载数据卷（主机目录:容器目录）
docker run -d -v /host/data:/container/data --name my_app <镜像名>

# 运行容器并设置环境变量
docker run -d -e "ENV_VAR=value" --name my_app <镜像名>
```

### 1.3 查看

```bash
# 查看正在运行的容器
docker ps

# 查看所有容器（包括已停止的）
docker ps -a

# 查看最近创建的容器
docker ps -l


# 列出本机正在运行的容器
docker container ls

# 列出本机所有容器，包括终止运行的容器
docker container ls --all

# run 总是新建容器，start 启动存在的容器
docker container start
```



查看源码：
```bash
# 容器内源码地址
docker run -it --rm --entrypoint /bin/sh thecharlesblake/solvitaire:1.0

# 新起shell, 查看容器id
docker ps

# 将 /bin/sh 复制到本地
docker cp 836b933a810c:/home/Solvitaire ./empty
```



### 1.4 源


正常配置位置：/etc/docker/daemon.json

```bash
# 创建或编辑 Docker daemon 配置
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
    "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
    ]
}
EOF

# 重启 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker
```

> 对应客户端设置中的 Docker Engine, 直接新增 registry-mirrors 项即可




## 制作 Docker 容器

```bash
git clone https://github.com/ruanyf/koa-demos.git
cd koa-demos
```



**1. 编写 Dockerfile**

.dockerignore 包含被忽视的文件

```txt
.git
node_modules
npm-debug.log
```

简易的 Dockerfile：

```bash
# 指定基础镜像
FROM node:8.4

# 将当前目录下的所有文件复制到
COPY . /app

# 指定工作路径，后续指令会在这目录执行
WORKDIR /app

# 设置环境变量为当前目录
ENV PYTHONPATH=.

# 运行 npm install 安装依赖。安装后所有的依赖，都打包到 image
RUN npm install --registry=https://registry.npm.taobao.org

# 允许外部连接这个端口
EXPOSE 3000

CMD node demos/01.js
```

**2. 创建 image**

```bash
docker image build -t koa-demo .

docker image ls
```

**3. 生成容器**

```bash
# 从 image 文件生成容器

docker container run -p 8000:3000 -it koa-demo /bin/bash
```


## Docker Compose

> 管理多个容器的联动

定义配置文件 docker-compose.yml

```bash
# up 启动服务，-d 表示 detached 模式，在后台运行不阻塞当前的命令行
# --build 启动容器之前重新构建服务的镜像
docker-compose up -d --build

docker-compose stop
```


## other


**Vagrant 和 Docker的使用场景和区别?**

|            | Vagrant                                        | Docker                                  |
| ---------- | ---------------------------------------------- | --------------------------------------- |
| 核心理念   | 提供一致的虚拟机开发环境（基于 VirtualBox 等） | 提供轻量级的容器环境（基于容器虚拟化）  |
| 底层技术   | 使用虚拟机（VirtualBox、VMware、Hyper-V）      | 使用容器（Linux 内核命名空间、cgroups） |
| 启动速度   | 慢（完整虚拟机）                               | 快（容器共享主机内核）                  |
| 系统隔离性 | 完整操作系统，隔离性强                         | 共享主机内核，隔离性较弱（但轻量）      |


| 使用场景                             | 选择 Vagrant             | 选择 Docker                                  |
| ------------------------------------ | ------------------------ | -------------------------------------------- |
| 开发跨平台项目（如 Windows + Linux） | ✅ 更适合                 | ⛔ Windows 容器支持较差                       |
| 模拟生产环境（如完整 OS）            | ✅ 适合                   | ⛔ 容器不是完整操作系统                       |
| 构建和部署微服务应用                 | ⛔ 过重                   | ✅ 非常适合（如 Docker Compose + Kubernetes） |
| 快速迭代和自动化 CI/CD 流程          | ⛔ 不够灵活               | ✅ 容器启动快，易集成                         |
| 操作系统内核级别开发、内核模块测试   | ✅ 虚拟机可以定制内核     | ⛔ 容器不支持更换内核                         |
| 学习/测试不同版本的 Linux 发行版     | ✅ 每个 VM 可安装不同系统 | ⛔ 容器一般依赖宿主机内核                     |
| 资源占用和启动速度要求较高           | ⛔ 虚拟机重、慢           | ✅ 容器轻量、快                               |



- 如果你正在开发微服务应用、注重部署效率和资源利用，首选 Docker。
- 如果你需要模拟完整的 OS 环境或运行多平台测试，选择 Vagrant 更合适。
- 总结：Vagrant是“环境即代码”，Docker是“应用即代码”。根据隔离需求和性能要求权衡选择。




--------------

参考资料：
- [Docker介绍.知乎](https://zhuanlan.zhihu.com/p/187505981)
- [阮一峰 Docker 入门教程](https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
- [阮一峰 Docker 微服务教程](https://www.ruanyifeng.com/blog/2018/02/docker-wordpress-tutorial.html)
- https://www.runoob.com/docker/docker-tutorial.html
- gpt