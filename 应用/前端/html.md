`html`：超文本 **标记** 语言，html文档也叫做Web页面。

**html基本语法：**

```html
<!DOCTYPE html>
<html>
    <head>
        <!-- 所有连接标签的默认链接 -->
        <base href="" target="_blank">
        <link rel="stylesheet" type="text/css" href="mystyle.css">
        <meta charset="utf-8">
        <meta name="keywords" content="">
        <meta name="description" content="">
        <!-- 定义网页编码格式为utf-8 -->

        <title>hello,html</title>
    </head>
    <body>
        <h1>这是一个标题</h1>
        <!-- 标题通过h1~h6标签定义 -->

        <!-- 使用img来更改图片大小 -->
        <img src="图片路径" style="zoom:50%">

        <p>这是一个段落</p>
        <a href="www.baidu.com">这是一个链接</a>

        <img loading="lazy" src="" width="258" height="39" />

        <br />
        <!-- br表示换行 --> 
        <hr />
        <!-- hr创建水平线 --> 
    </body>
</html>
```

**html特殊语法：**

```html
<!-- 无序列表，有序列表以o开头 -->
<ul>
    <li>项目</li>
    <li>项目</li>
</ul>

<!-- 表单，密码字段 -->
<form>
Password: <input type="password" name="pwd">
</form>

<!-- 框架，用于显示另一个页面 -->
<iframe loading="lazy" src="demo_iframe.htm" width="200" height="200"></iframe>
```

**other:**

- 属性（可以在元素中附加信息）
  - class
  - id
  - style
  - title
- 实体
  - HTML 中，某些字符是预留的	`&lt;` 等等
- 使用脚本


```html
<script>
function myFunction()
{
	document.getElementById("demo").innerHTML="Hello JavaScript!";
}
</script>

<button type="button" onclick="myFunction()">点我</button>
```

[html 速查列表](https://www.runoob.com/html/html-quicklist.html)

```html
<form action="demo-form.php" autocomplete="on">
  First name:<input type="text" name="fname"><br>
  Last name: <input type="text" name="lname"><br>
  E-mail: <input type="email" name="email" autocomplete="off"><br>
  <input type="submit">
</form>
```

html5 新元素：
- 更多的 input 类型
- 表单属性、表达你元素
- web 存储， 一种比 cookie 更好的本地存储方式
  - localStorage，长久保存
  - sessionStorage，关闭窗口或标签后删除
- web socket