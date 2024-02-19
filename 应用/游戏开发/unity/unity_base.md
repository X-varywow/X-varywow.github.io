## _简介_


好久没看 unity 了，unitypackage 几乎是一切的打包，都忘了。


</br>

## _Prefab_

在Unity中，Prefab是一种可重复使用的游戏对象模板。它们是预先定义好的游戏对象，可以在场景中多次实例化。

Prefab的主要作用是将游戏对象的属性、组件和子对象保存为一个模板。通过使用Prefab，可以快速创建多个相似的游戏对象，而不需要每次都手动设置它们的属性。

使用Prefab的步骤如下：

1. 创建一个游戏对象，并设置其属性、组件和子对象。
2. 将该游戏对象拖拽到Project窗口中，以创建一个Prefab。
3. 在需要使用该Prefab的场景中，将Prefab拖拽到Hierarchy或Scene视图中，即可在场景中实例化该Prefab。
4. 在代码中，可以通过Instantiate函数动态创建Prefab的实例，并对其进行操作。

Prefab的使用可以大大简化游戏对象的创建和管理，提高开发效率。同时，通过修改Prefab，可以一次性修改多个实例的属性和组件，提供了方便的批量操作功能。


将 controller 添加到 prefab 中，想起来了，这个 controller 会给 prefab object 添加一些可调整的参数。 unity 还是好玩的


</br>

## _other_


https://github.com/homuler/MediaPipeUnityPlugin

那似乎没有 python 什么事了


----------------


[unity 中使用 torch](https://medium.com/@a.abelhopereira/how-to-use-pytorch-models-in-unity-aa1e964d3374)

using TorchSharp

https://github.com/Unity-Technologies?q=&type=all&language=&sort=stargazers


-----------------


[unity调用python脚本](https://juejin.cn/s/unity%E8%B0%83%E7%94%A8python%E8%84%9A%E6%9C%AC)

unity + python 的几种方式：
- [unity python scripting](https://docs.unity3d.com/Packages/com.unity.scripting.python@6.0/manual/index.html)
- 两端运行
- unity 中创建进程 ProcessStartInfo start = new ProcessStartInfo();
