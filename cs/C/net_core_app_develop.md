

使用 c # 语言

visual studio 2022

.NET 8.0 框架


</br>

## _控制台应用程序_


```csharp
// See https://aka.ms/new-console-template for more information
namespace HelloWorld;
internal class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}
```


</br>

## _WPF 应用程序_


自动创建的文件结构：
- project_name
  - bin
  - obj
  - App.xaml, App.xaml.cs
  - MainWindow.xaml, MainWindow.xaml.cs
  - csproj
- project_name.sln


> solution file, 用于 vs 中组织和管理项目的容器，通常包含：配置信息、项目引用、构建选项、调试设置等



```xaml
<Window x:Class="WpfApp1.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:d="http://schemas.microsoft.com/expression/blend/2008"
        xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
        xmlns:local="clr-namespace:WpfApp1"
        mc:Ignorable="d"
        Title="WPF Application" Height="618" Width="1000">
    <Grid Margin="10">

        <Grid.RowDefinitions>
            <RowDefinition Height="Auto" />
            <RowDefinition Height="*" />
        </Grid.RowDefinitions>

        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="*" />
            <ColumnDefinition Width="*" />
        </Grid.ColumnDefinitions>

        <Label>Names</Label>
        <ListBox Grid.Row="1" x:Name="lstNames" />
        <StackPanel Grid.Row="1" Grid.Column="1" Margin="5,0,0,0">
            <TextBox x:Name="txtName" />
            <Button x:Name="btnAdd" Margin="0,5,0,0" Click="ButtonAddName_Click">Add Name</Button>
        </StackPanel>
    </Grid>
</Window>
```

x:name 为控件绑定变量

Click 为其绑定函数事件

```csharp
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void ButtonAddName_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrWhiteSpace(txtName.Text) && !lstNames.Items.Contains(txtName.Text))
            {
                lstNames.Items.Add(txtName.Text);
                txtName.Clear();
            }

        }
    }
}
```


可以的，图形界面开发不算慢。

</br>

## _更多控件&布局_

https://learn.microsoft.com/zh-cn/dotnet/desktop/wpf/overview/?view=netdesktop-8.0#controls


开始 [.NET 桌面应用程序](https://dotnet.microsoft.com/zh-cn/apps/desktop) WinUI3

- .NET MAUI (c# xaml 开发桌面应用)
- Blazor (使用 c# html css 实现跨桌面 web)
- WinUI 和 WinAppSDK


---------------

参加资料：
- [教程：使用 Visual Studio 创建 .NET 控制台应用程序](https://learn.microsoft.com/zh-cn/dotnet/core/tutorials/with-visual-studio?pivots=dotnet-8-0)
- [教程：使用 .NET 创建新的 WPF 应用](https://learn.microsoft.com/zh-cn/dotnet/desktop/wpf/get-started/create-app-visual-studio?view=netdesktop-8.0)