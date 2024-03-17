

## preface


为什么 c#?

作为 python 的补充，开发 unity 会用到，并且以前使用外挂时用的 c#，.NET 也很有用，对 windows 可有更多的 DIY


> .NET 是一个安全、可靠且高性能的应用程序平台。</br>
C# 是 .NET 的编程语言。它是强类型且类型安全的，并集成了并发和自动内存管理。</br>
C# 是一种新式、安全且面向对象的编程语言，既有面向数据的记录等高级功能，也有函数指针等低级功能。


.NET 包括一组标准库和 API，涵盖集合、网络到机器学习。

NuGet 是 .NET 的包管理器

## hello world

C# 程序在 .NET 上运行；[下载 net sdk](https://dotnet.microsoft.com/zh-cn/download/dotnet/sdk-for-vs-code)

```bash
dotnet --version
# 8.0.203
```


```cs
using System;

class Hello{
    static void Main(){
        Console.WriteLine("Hello, world");
    }
}
```

## 类型变量

.net.dib, vscode 内类似 jupyter


```cs
string aFriend = "Bill";
Console.WriteLine(aFriend);

Console.WriteLine("Hello " + aFriend);

Console.WriteLine($"My friends are {firstFriend}");


var apples = 100m;   // Decimal value
var oranges = 30m;   // Decimal value

display(apples > oranges)
```

## 循环控制

```cs
var seconds = DateTime.Now.Second;
display("Current seconds: " + seconds);

if (seconds % 2 == 0) {
    display("Seconds are even");
} else if (seconds % 3 == 0) {
    display("Seconds are a multiple of 3");
} else if (seconds % 5 == 0) {
    display("Seconds are a multiple of 5");
} else {
    display("Seconds are neither even nor a multiple of 3");
}

if (seconds % 2 == 0)          display("Seconds are even");
else if (seconds % 3 == 0)     display("Seconds are a multiple of 3");
else if (seconds % 5 == 0)     display("Seconds are a multiple of 5");
else                           display("Seconds are neither even nor a multiple of 3");
```

```cs
for (var counter=5; counter>0; counter-= 3) {
  display("Counting " + counter);
}
```

```cs
var arrNames = new string[] { "Fritz", "Scott", "Maria", "Jayme", "Maira", "James"};

foreach (var name in arrNames) {
    display(name);
}
```

```cs
var counter = 6;

while (counter < 5) {
    counter++;
    display(counter);
}
```


```cs
var counter = 6;

do {
    counter++;
    display(counter);
} while (counter < 5);
```





## 示例项目

[虚拟桌宠模拟器](https://github.com/LorisYounger/VPet)

[Game Launcher for miHoYo - 米家游戏启动器](https://github.com/Scighost/Starward)


## desktop runtime


.NET windows desktop



--------------

参考资料：
- https://learn.microsoft.com/zh-cn/dotnet/csharp/tour-of-csharp/
- https://dotnet.microsoft.com/zh-cn/learn
- https://dotnet.microsoft.com/zh-cn/learntocode


