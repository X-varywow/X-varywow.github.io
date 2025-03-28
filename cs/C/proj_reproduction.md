

记录一个 c 项目 的复现过程

https://github.com/thecharlieblake/Solvitaire


## docs

CMake（Cross Platform Make）是一个跨平台的 **构建系统**，用于管理软件的编译、测试和打包。它主要用于生成原生的构建系统（如 Makefile、Ninja、Visual Studio 解决方案等），而不是直接进行编译。


vcpkg 是 Microsoft 和 C++ 社区维护的免费开放源代码 C/C++ 包管理器, 于 2016 年推出

安装方法：
- https://github.com/Microsoft/vcpkg
- 下载后运行 bat
- bat 下载一个 exe, 窗口卡，换用浏览器下
- 添加 exe 所在文件夹到系统路径



## 安装

```bash
cd C:\Users\Administrator\Documents\vcpkg


vcpkg install boost-program-options rapidjson
```


去你的吧，gpt 错误答案，网上全复制粘贴几年前的，加速源没一个，vcpkg 就是个重试模拟器。


----------


用 docker 了，

windows docker wsl 下载系统要几个小时


mac:

```bash
docker pull thecharlesblake/solvitaire:1.0


# 跑一个解牌示例
./enter-container.sh "./solvitaire --type klondike --random 1"

# 进入 bash
docker run --rm -it thecharlesblake/solvitaire:1.0 bash

# cmake 版本报错，将脚本镜像换到另一个项目; 好像没啥用
./enter-container.sh ./build.sh
```


```bash
./enter-container.sh

./solvitaire --type klondike --random 1


./solvitaire --type klondike src/test/resources/klondike/*.json > output_g1.txt

./solvitaire --type klondike src/test/resources/klondike/test.json > output_g1.txt

./solvitaire --type klondike --streamliners both --timeout 20000 src/test/resources/klondike/test.json > output_g1.txt
```



用的另一个镜像， dockerfile:

```bash
FROM ubuntu:18.04

WORKDIR /home/Solvitaire/

COPY . /home/Solvitaire/

RUN apt-get update \
    && apt-get install -y \
    gcc-6 \
    cmake \
    libboost-all-dev \
    git \
    python3 \
    vim \
    parallel \
    wget

RUN apt-get remove -y cmake

RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.tar.gz \
    && tar -xzvf cmake-3.25.1-linux-x86_64.tar.gz --strip-components=1 -C /usr/local \
    && rm cmake-3.25.1-linux-x86_64.tar.gz

RUN cmake --version

RUN cmake -DCMAKE_BUILD_TYPE=RELEASE -Bcmake-build-release -H. \
    && cmake -DCMAKE_BUILD_TYPE=DEBUG -Bcmake-build-debug -H.

RUN cmake --build cmake-build-release \
    && cmake --build cmake-build-debug

WORKDIR /home/Solvitaire/

CMD ["/bin/bash"]
```






## replay


整体思想：

深度优先搜索（DFS）：Solvitaire的核心算法是深度优先搜索，通过回溯法遍历所有可能的游戏状态，直到找到解或证明无解。

置换表（Transposition Tables）：用于避免重复搜索相同的游戏状态，减少搜索空间。

对称性（Symmetry）：通过识别对称的游戏状态，进一步减少搜索空间。

支配规则（Dominances）：通过支配规则，避免搜索那些必然不会导致更好解的路径。例如，某些情况下可以强制将牌移动到基础堆（foundation），而不需要回溯。

Streamliners：通过引入额外的约束条件（如强制将牌移动到基础堆），减少搜索空间，虽然这可能导致一些解被忽略，但可以显著提高求解速度。

Trailing, 回溯的一种优化，通过记录变量的修改历史，在回溯时只恢复被修改的部分，而不是整个状态，从而提高性能。