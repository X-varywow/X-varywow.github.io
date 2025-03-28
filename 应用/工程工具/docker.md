

容器，用于实现环境打包隔离。

Docker 是一个用 Go语言实现的开源项目，对 Linux 容器的一种封装（虚拟化），提供简单易用的容器使用接口。


</br>

## _基础_

- dockerfile（配置文件）
- image （镜像文件，包含应用程序及其依赖）
- container (容器实例) (**拉取 image 创建 container 实例**)


```bash
# 将 dockerfile 交给 docker 编译，之后会创建可执行程序 image
docker build

# 将 image 加载到内存，作为一个 container 运行起来
docker run

# 列出所有运行过的容器，包括正在运行和已经停止的容器。
docker ps -a
```

</br>

demo: hello world

```bash
docker image pull library/hello-world

docker image ls

docker container run hello-world
```

> run 会比 runc 多出自动拉取、创建容器的步骤


</br>

其它命令：

```bash
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





</br>

## _制作 Docker 容器_

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

</br>

## _Docker Compose_

> 管理多个容器的联动

定义配置文件 docker-compose.yml

```bash
# up 启动服务，-d 表示 detached 模式，在后台运行不阻塞当前的命令行
# --build 启动容器之前重新构建服务的镜像
docker-compose up -d --build

docker-compose stop
```




--------------

参考资料：
- [Docker介绍.知乎](https://zhuanlan.zhihu.com/p/187505981)
- [阮一峰 Docker 入门教程](https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
- [阮一峰 Docker 微服务教程](https://www.ruanyifeng.com/blog/2018/02/docker-wordpress-tutorial.html)
- https://www.runoob.com/docker/docker-tutorial.html