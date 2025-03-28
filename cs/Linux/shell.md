
>shell: 提供了用户与内核交互的接口，指 UNIX/Linux 的命令解释器

常见的shell:
- sh, unix 的默认 shell
- bash, linux 的默认 shell
- zsh

shell 语言属于脚本语言，需要通过解释执行。

```bash
# 查看当前 shell
echo $SHELL

# shell 进程名称 
echo $0
```



## shell基础

### 1.1 简单样例

```bash
#!/bin/bash

echo "What is your name?"
read PERSON
echo "Hello, $PERSON"
```

`#!` 用于声明解释脚本所用的解释器



变量是一个名字，不是一个值，用`$`来取值

| 指令      | 描述                              |
| --------- | --------------------------------- |
| `echo`    | 打印一行，自动换行                |
| `printf`  |                                   |
| `var=num` | 变量不区分类型，`=`前后不能有空格 |

-----------

### 1.2 运行shell脚本

（方式一：作为可执行程序）
```bash
#使脚本具有执行权限
chmod +x ./test.sh
# 执行脚本
./test.sh
```

> 为什么需要 ./webui.sh 来执行sh 文件，而直接 webui.sh 不行?</br></br>
> 在命令行中执行脚本文件时，需要指定脚本文件的路径。当使用 `./webui.sh` 时,表示当前目录下的 `webui.sh` 文件。</br>
> 如果直接输入 `webui.sh`，系统会在环境变量 `$PATH` 所指定的路径中去查找可执行文件。如果 `$PATH` 中没有包含当前目录 `.`，那么系统就无法找到 `webui.sh` 文件并执行。</br>
> 因此，为了确保能够执行脚本文件，需要提供文件的路径，即使用 `./webui.sh` 这种形式来执行脚本文件。


（方式二：作为解释器参数）
```bash
sh test.sh
```


### 1.3 运算符

关系运算符只支持数字，不支持字符串，除非字符串的值是数字。

| 运算符      | 说明             |
| ----------- | ---------------- |
| `-eq`       | 两数相等返回true |
| `-ne`       |                  |
| `-gt` `-ge` | `>` `>=`         |
| `-lt` `-le` | `<` `<=`         |

```bash
num1=100
num2=100
if test $[num1] -eq $[num2]
then
    echo '两个数相等！'
else
    echo '两个数不相等！'
fi
```

```bash
#!/bin/bash

a=5
b=6

result=$[a+b] # 注意等号两边不能有空格
echo "result 为： $result"
```

### 1.4 流程控制

```bash
a=10
b=20
if [ $a == $b ]
then
   echo "a 等于 b"
elif [ $a -gt $b ]
then
   echo "a 大于 b"
elif [ $a -lt $b ]
then
   echo "a 小于 b"
else
   echo "没有符合的条件"
fi
```


```bash
num1=$[2*3]
num2=$[1+5]
if test $[num1] -eq $[num2]
then
    echo '两个数字相等!'
else
    echo '两个数字不相等!'
fi
```

```bash
for loop in 1 2 3 4 5
do
    echo "The value is: $loop"
done
```


```bash
#!/bin/bash
int=1
while(( $int<=5 ))
do
    echo $int
    let "int++"
done
```


```bash
echo '按下 <CTRL-D> 退出'
echo -n '输入你最喜欢的网站名: '
while read FILM
do
    echo "是的！$FILM 是一个好网站"
done
```


```bash
#!/bin/bash

a=0

until [ ! $a -lt 10 ]
do
   echo $a
   a=`expr $a + 1`
done
```


```bash
echo '输入 1 到 4 之间的数字:'
echo '你输入的数字为:'
read aNum
case $aNum in
    1)  echo '你选择了 1'
    ;;
    2)  echo '你选择了 2'
    ;;
    3)  echo '你选择了 3'
    ;;
    4)  echo '你选择了 4'
    ;;
    *)  echo '你没有输入 1 到 4 之间的数字'
    ;;
esac
```


```bash
#!/bin/bash
while :
do
    echo -n "输入 1 到 5 之间的数字:"
    read aNum
    case $aNum in
        1|2|3|4|5) echo "你输入的数字为 $aNum!"
        ;;
        *) echo "你输入的数字不是 1 到 5 之间的! 游戏结束"
            break
        ;;
    esac
done
```


```bash
#!/bin/bash
while :
do
    echo -n "输入 1 到 5 之间的数字: "
    read aNum
    case $aNum in
        1|2|3|4|5) echo "你输入的数字为 $aNum!"
        ;;
        *) echo "你输入的数字不是 1 到 5 之间的!"
            continue
            echo "游戏结束"
        ;;
    esac
done
```

### 1.5 函数

```bash
#!/bin/bash

funWithReturn(){
    echo "这个函数会对输入的两个数字进行相加运算..."
    echo "输入第一个数字: "
    read aNum
    echo "输入第二个数字: "
    read anotherNum
    echo "两个数字分别为 $aNum 和 $anotherNum !"
    return $(($aNum+$anotherNum))
}
funWithReturn
echo "输入的两个数字之和为 $? !"
```

### 1.6 重定向

| 命令            | 说明               |
| --------------- | ------------------ |
| command > file  | 输出重定向（覆盖） |
| command < file  | 输入重定向         |
| command >> file | 输出重定向（追加） |


tee 可以同时将输出打印到标准输出和文件：
```bash
echo "这是一条信息" | tee -a /var/log/jupyter.log
```


### 1.7 代码封装&引用

```bash
# test.sh

url = "123"
```

```bash
# test2.sh
# 使用 . 进行引用，注意：带有空格

. ./test.sh

echo "$url"
```




## demo

demo. 猜数字

注：shell中，可以使用$RANDOM获得一个随机整数。

```bash
cd ~
echo "some info"
read -p "input ceil num:" ceil
x = $[RANDOM%ceil+1]
read -p "input guess num:" num
while [ "$num" != "$x" ]
do
    if [ "$num" -lt "$x"]
    then
        echo "your number is small"
        read -p "input again" num
    else
        echo "your number is big"
        read -p "input again" num
    fi
done
echo "succeed"
```

**注意：** 
- `[` `]`左右要有空格
- `"$num"`，`"$x"` 的双引号都可以去掉
- shell算术运算符有`==`，没有`>` `<`


-----------

demo. enter docker container

```bash
#!/bin/bash

# 判断脚本参数个数
if [ "$#" == 0 ]; then
    # 将当前目录挂载到容器的/home/Solvitaire
    # --rm ：容器退出后自动删除容器
    docker run -it -v$(pwd):/home/Solvitaire --rm \
    joshstoneback/modified-solvitaire:latest
else
    docker run -it -v$(pwd):/home/Solvitaire --rm \
    joshstoneback/modified-solvitaire:latest bash -c "$*"
fi
```







-----------

一般情况下 vim建立的文本文件属性值为`644`，需要添加`x`属性 `chmod 755 ~~~`


## 构建自动化流程


参考 [stable-diffusion-webui.sh](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/webui.sh)

`-z` 判断字符串的长度是否为 0



## bash快捷键

| 快捷键   | 描述                         |
| -------- | ---------------------------- |
| Ctrl + a | 移动光标到行首               |
| Ctrl + e | 移动光标到行尾               |
| Alt + b  | 移动光标后退一个单词（词首） |
| Alt + f  | 移动光标前进一个单词（词首） |
| Ctrl + l | 清屏                         |


[熟悉 Bash 快捷键来提高效率](https://harttle.land/2015/11/09/bash-shortcuts.html)



## other

`set -e` 设置脚本在遇到任何错误时立即退出




----------------

参考资料：
- [shell 概述](http://kuanghy.github.io/shell-tutorial/chapter1.html)
- https://www.runoob.com/linux/linux-shell.html


