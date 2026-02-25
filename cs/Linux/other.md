

sshfs 磁盘挂载

tcpdump 查看某个网站的 TCP 通信



</br>

## _文件打包_

```bash
# 仅打包，不压缩
tar -cvf archive_name.tar /path/to/directory

# -z 表示使用 gzip 压缩
tar -czvf archive_name.tar.gz /path/to/directory

# zip
zip -r archive_name.zip /path/to/directory
```

keyword: linux 打包 zip

</br>

## _定时任务_

`crontab` 用于设置周期性执行指令， cron 是一个守护进程




</br>

## _aws安装lfs_

```bash
which amazon-linux-extras

sudo yum install -y amazon-linux-extras
amazon-linux-extras list

sudo amazon-linux-extras install epel -y 
sudo yum-config-manager --enable epel
sudo yum install git-lfs -y

# after，git will be auto git lfs, you can obvious feel the difference
```

[参考](https://stackoverflow.com/questions/71448559/git-large-file-storage-how-to-install-git-lfs-on-aws-ec2-linux-2-no-package)


</br>

## _统计文件夹下的文件数目_


```bash
# 统计当前目录下文件的个数（不包括目录）
ls -l | grep "^-" | wc -l

# 统计当前目录下文件的个数（包括子目录）
ls -lR| grep "^-" | wc -l

# 查看某目录下文件夹(目录)的个数（包括子目录）
ls -lR | grep "^d" | wc -l
```

[参考](http://noahsnail.com/2017/02/07/2017-02-07-Linux%E7%BB%9F%E8%AE%A1%E6%96%87%E4%BB%B6%E5%A4%B9%E4%B8%8B%E7%9A%84%E6%96%87%E4%BB%B6%E6%95%B0%E7%9B%AE/)




</br>

## _安装 ffmpeg_

```bash
# 查看内核版本
cat /proc/version

git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
./configure --prefix=/usr/local/ffmpeg --disable-x86asm
sudo make && sudo make install

sudo vi /etc/profile

# 在最后PATH添加环境变量：export PATH=$PATH:/usr/local/ffmpeg/bin
# 保存退出

source /etc/profile
ffmpeg -version
```

```bash
# 可以 更换地址，这样就不必每次sagemaker重启都编译了(/usr/local 每次重启会被清空)

./configure --prefix=/home/ec2-user/SageMaker/ffmpeg_maked --disable-x86asm --enable-libx264
# export PATH=$PATH:/home/ec2-user/SageMaker/ffmpeg_maked/bin
```

sagemaker 中系统没有自带的 x264，（通常以 dll 缺失的报错出现），需进行如下操作：

1. 安装编译工具和依赖库：
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake mercurial yasm
```

2. 下载x264源代码：
```bash
git clone https://bitbucket.org/multicoreware/x264
cd x264
```

3. 编译和安装x264库：
```bash
./configure --prefix=/usr/local/x264 --enable-shared --disable-asm
sudo make && sudo make install

which x264
# /usr/local/bin/x264
```

4. 安装pkg-config：
```bash
sudo yum install pkgconfig
```

5. 将x264库路径添加到pkg-config的搜索路径中：
```bash
export PKG_CONFIG_PATH=/usr/local/x264/lib/pkgconfig:$PKG_CONFIG_PATH
```

6. 然后重新运行configure命令：
```bash
./configure --prefix=/home/ec2-user/SageMaker/ffmpeg_maked --disable-x86asm --enable-libx264 --enable-gpl
```


---------

顺带解释一下编译构建过程，及各种中间文件：

- `CC` C Compiler, C编译器，将源代码翻译成机器语言
- `AR` Archiver, 静态库文件生成工具
- `GEN` Generator, 代码生成器


---------

- `.so` 文件，Shared Object
  - 它是一种在运行时加载的库文件，可以被多个程序共享使用。
- `.a` 文件，Static Library
  - 静态库文件包含了可重定位的目标文件，以及用于链接的元数据信息。静态库在编译时被链接到程序中，成为程序的一部分
- `.o` 文件，Object File
  - 它是编译器将源代码编译成机器代码后生成的中间文件。目标文件包含了二进制代码和相关的符号表信息，但它还没有被链接器处理，无法直接执行。
- `lib`（静态链接库）
  - 在编译时被完全复制到程序中，程序运行时不再依赖lib文件的存在
- `dll`（动态链接库）
  - 当程序使用dll文件时，链接器只会在可执行文件中包含对dll文件的引用，而不会将dll文件的代码和数据复制到可执行文件中。
  - windows 系统特有的，允许程序运行时动态加载； linux 系统中类似概念为 共享库（.so）