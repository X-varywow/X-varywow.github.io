

## preface


为什么学 c#?

作为 python 的补充，开发 unity 会用到，并且以前使用外挂时用的 c#，.NET 也很有用，对 windows 可有更多的 DIY 特性。


## hello world

C# 程序在 .NET 上运行；

.NET 是名为公共语言运行时 (CLR) 的虚执行系统和一组类库。 

CLR 是 Microsoft 对公共语言基础结构 (CLI) 国际标准的实现。 

CLI 是创建执行和开发环境的基础，语言和库可以在其中无缝地协同工作。


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





## 项目

[虚拟桌宠模拟器](https://github.com/LorisYounger/VPet)

[Game Launcher for miHoYo - 米家游戏启动器](https://github.com/Scighost/Starward)


## desktop runtime


.NET windows desktop



--------------

参考资料：
- https://learn.microsoft.com/zh-cn/dotnet/csharp/tour-of-csharp/
- https://dotnet.microsoft.com/zh-cn/learn
- https://dotnet.microsoft.com/zh-cn/learntocode


