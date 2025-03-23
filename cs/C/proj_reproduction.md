

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

docker wsl 下载系统要几个小时


## python 重写