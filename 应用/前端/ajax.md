AJAX 是一种在无需重新加载整个网页的情况下，能够更新部分网页的技术

AJAX = 异步 JavaScript 和 XML
.
AJAX 是一种用于创建快速动态网页的技术，通过在后台与服务器进行少量数据交互，可以使网页实现异步更新，这意味着可以在不重新加载整个网页的情况下，对网页的某部分更新。

- 使用的技术
  - XMLHttpRequest, 异步地与服务器交换数据
  - JavaScript/DOM， 信息显示，交互
  - CSS
  - XML, 作为转换数据的格式



>示例1：为 button 绑定一个点击事件，然后从服务器传来一个东西

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script>
function loadXMLDoc()
{
	var xmlhttp;
	if (window.XMLHttpRequest)
	{
		//  IE7+, Firefox, Chrome, Opera, Safari 浏览器执行代码
		xmlhttp=new XMLHttpRequest();
	}
	else
	{
		// IE6, IE5 浏览器执行代码
		xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
	}
	xmlhttp.onreadystatechange=function()
	{
		if (xmlhttp.readyState==4 && xmlhttp.status==200)
		{
			document.getElementById("myDiv").innerHTML=xmlhttp.responseText;
		}
	}
	xmlhttp.open("GET","/try/ajax/ajax_info.txt",true);
	xmlhttp.send();
}
</script>
</head>
<body>

<div id="myDiv"><h2>使用 AJAX 修改该文本内容</h2></div>
<button type="button" onclick="loadXMLDoc()">修改内容</button>

</body>
</html>
```

>.readyState==4 表示请求已完成，且响应已就绪
>.status==200 表示 OK

`xmlhttp=new XMLHttpRequest();` 创建一个请求（通用）；

```html
xmlhttp.open("GET","/try/ajax/demo_get.php",true);
xmlhttp.send();

xmlhttp.open("GET","/try/ajax/demo_get.php?t=" + Math.random(),true);
xmlhttp.send();

<!--如果来自服务器的响应并非 XML，请使用 responseText 属性。-->

document.getElementById("myDiv").innerHTML=xmlhttp.responseText;
```

[如何在输入时与服务器通信交互](https://www.runoob.com/try/try.php?filename=tryajax_suggest)

[用 AJAX 从数据库返回数据](https://www.runoob.com/try/try.php?filename=tryajax_database)


>推荐使用 jquery 实现 AJAX: