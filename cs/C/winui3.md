


```bash
winget install "Visual Studio Community 2022"  --override "--add Microsoft.VisualStudio.Workload.ManagedDesktop Microsoft.VisualStudio.ComponentGroup.WindowsAppSDK.Cs" -s msstore
```

https://github.com/microsoft/WindowsAppSDK-Samples





-------------


MSIX 是一种 Windows 应用包格式, WINDOWS 的打包工具。[什么是 MSIX?](https://learn.microsoft.com/zh-cn/windows/msix/overview)


直接生成 -> 打包

文件位置：bin\x64\Debug\net6.0-windows10.0.19041.0\win10-x64\AppPackages\DeploymentManagerSample_1.1.0.0_x64_Debug_Test

无法验证发布者证书，安装不了

不如 electron, 不借用 c# 特性的话

--------------

参考资料：
- [创建第一个 WinUI 3（Windows 应用 SDK）项目](https://learn.microsoft.com/zh-cn/windows/apps/winui/winui3/create-your-first-winui3-app)
- [教程：使用 WinUI 3 创建简单的照片查看器](https://learn.microsoft.com/zh-cn/windows/apps/get-started/simple-photo-viewer-winui3)