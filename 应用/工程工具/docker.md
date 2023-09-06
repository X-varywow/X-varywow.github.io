

容器，轻量级实现环境打包隔离。

Docker 是一个用 Go语言实现的开源项目，

Docker 属于 Linux 容器的一种封装，提供简单易用的容器使用接口。


## 基础


- dockerfile（配置文件）
- image （镜像文件，包含应用程序及其依赖）
- container (容器实例)


```bash
# 将 dockerfile 交给 docker 编译，之后会创建可执行程序 image
docker build

# 将 image 加载到内存，作为一个 container 运行起来
docker run

# 列出所有运行过的容器，包括正在运行和已经停止的容器。
docker ps -a
```


## 实例：hello world

```bash
docker image pull library/hello-world

docker image ls

docker container runc hello-world
```

## 制作 Docker 容器

```bash
git clone https://github.com/ruanyf/koa-demos.git
cd koa-demos
```



### 编写 Dockerfile

.dockerignore 包含被忽视的文件

```txt
.git
node_modules
npm-debug.log
```

简易的 Dockerfile：

```bash
FROM node:8.4

# 将当前目录下的所有文件
COPY . /app

# 指定工作路径
WORKDIR /app

# 运行 npm install 安装依赖。安装后所有的依赖，都打包到 image
RUN npm install --registry=https://registry.npm.taobao.org

# 允许外部连接这个端口
EXPOSE 3000

CMD node demos/01.js
```

### 创建 image

```bash
docker image build -t koa-demo .

docker image ls
```

### 生成容器

```bash
# 从 image 文件生成容器

docker container run -p 8000:3000 -it koa-demo /bin/bash
```

## Docker Compose

> 管理多个容器的联动

定义配置文件 docker-compose.yml

```bash
docker-compose up

docker-compose stop
```

## milvus 流程

```bash
Step 1/17 : FROM node:14.16-alpine as builder

Step 4/17 : RUN yarn
[1/4] Resolving packages...
[4/4] Building fresh packages... 69s


RUN apt-get update && apt-get install -y    python3         python3-pip     gunicorn3 

然后一直 apt -get
```


```bash
Step 1/10 : From ubuntu:bionic-20200219
```




--------------

参考资料：
- https://zhuanlan.zhihu.com/p/187505981
- [阮一峰 Docker 入门教程](https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html)
- [阮一峰 Docker 微服务教程](https://www.ruanyifeng.com/blog/2018/02/docker-wordpress-tutorial.html)
- https://www.runoob.com/docker/docker-tutorial.html