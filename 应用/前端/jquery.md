>封装了许多常用的 js 代码，[菜鸟教程](https://www.runoob.com/jquery/jquery-tutorial.html)

##### 示例1：hide

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"> 
<title>菜鸟教程(runoob.com)</title> 
<script src="https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js">
</script>
<script>
$(document).ready(function(){
  $("p").click(function(){
    $(this).hide();
  });
});
</script>
</head>
<body>
<p>如果你点我，我就会消失。</p>
<p>继续点我!</p>
<p>接着点我!</p>
</body>
</html>
```

hide() 和 show()

```js
//speed 参数规定隐藏/显示的速度，"slow","fast"，毫秒
//callback 参数为操作完成后执行的函数名称

$("#hide").click(function(){
  $("p").hide(speed, callback);
});
 
$("#show").click(function(){
  $("p").show(speed, callback);
});
```
toggle()：hide 和 show 的结合

```js
$("button").click(function(){
  $("p").toggle();
});
```

##### 示例2：实现图片上传时预览

```js
// html file-input 下面需要有一个 id 为 preview 的图片，src可以先不定

$('input[type="file"]').change(function(e) {
  reader = new FileReader();
  reader.readAsDataURL(e.target.files[0]);
  reader.onload = function() {
    $('#preview').attr('src', reader.result);
  };
});
```

##### 示例3：瀑布流布局实现

**瀑布流布局**：指多个高度不定的 div 无序浮动。使用 float:left 和 js 实现， 还可以直接调用框架实现。


##### 示例4：操作 CSS

```js
$("button").click(function(){
  $("h1,h2,p").addClass("blue");
  $("div").addClass("important");
});
```


##### 示例5：AJAX load() 提示信息

```js
$("button").click(function(){
  $("#div1").load("demo_test.txt",function(responseTxt,statusTxt,xhr){
    if(statusTxt=="success")
      alert("外部内容加载成功!");
    if(statusTxt=="error")
      alert("Error: "+xhr.status+": "+xhr.statusText);
  });
});
```

##### 示例6：发送一个 GET 请求并返回结果

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>菜鸟教程(runoob.com)</title>
<script src="https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js">
</script>
<script>
$(document).ready(function(){
	$("button").click(function(){
		$.get("/try/ajax/demo_test.php",function(data,status){
			alert("数据: " + data + "\n状态: " + status);
		});
	});
});
</script>
</head>
<body>

<button>发送一个 HTTP GET 请求并获取返回结果</button>

</body>
</html>
```

```php
<?php
echo '这是个从PHP文件中读取的数据。';
?>
```

>data 是服务器执行 php 之后 echo 返回的信息



##### 示例6：发送一个 POST 请求并返回内容

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>菜鸟教程(runoob.com)</title>
<script src="https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js">
</script>
<script>
$(document).ready(function(){
	$("button").click(function(){
		$.post("/try/ajax/demo_test_post.php",{
			name:"菜鸟教程",
			url:"http://www.runoob.com"
		},
		function(data,status){
			alert("数据: \n" + data + "\n状态: " + status);
		});
	});
});
</script>
</head>
<body>

<button>发送一个 HTTP POST 请求页面并获取返回内容</button>

</body>
</html>
```


```php
<?php
$name = isset($_POST['name']) ? htmlspecialchars($_POST['name']) : '';
$url = isset($_POST['url']) ? htmlspecialchars($_POST['url']) : '';
echo '网站名: ' . $name;
echo "\n";
echo 'URL 地址: ' .$url;
?>
```

>data 是服务器执行 php 之后 echo 返回的信息

- 选择器
  - 元素选择器 `$("p")`
  - id选择器 `$("#test")`
  - class选择器 `$(".test")`
  - 其它
    - $(this), 选取当前 HTML 元素
    - $("p.intro")，选取 class 为 intro 的 p
    - $("[href]")
    - $("tr:even")
- 隐藏/显示
  - hide
  - show
  - toggle
- 淡入/淡出
  - fadeIn
  - fadeOut
  - fadeToggle()
  - fadeTo()
- 滑动
  - slideDown
  - slideUp
  - slideToggle
- 动画
  - animate({params}, speed, callback)
- [链](https://www.runoob.com/jquery/jquery-chaining.html)
- [捕获, 操作DOM](https://www.runoob.com/jquery/jquery-dom-get.html)
  - text()，设置或返回所选元素的文本内容
  - html()，设置或返回所选元素的内容（包括 HTML 标记）
  - val()，设置或返回表单字段的值
  - attr()，获取属性值，参考示例2
- [添加元素](https://www.runoob.com/jquery/jquery-dom-add.html)
  - append()
  - prepend()
  - after()
  - before()
- 删除元素
  - remove(), 删除被选元素
  - empty()，删除被选元素的子元素
  - 过滤被删除的元素
    - $("p").remove(".italic");
- 操作 CSS
- [处理尺寸](https://www.runoob.com/jquery/jquery-dimensions.html)
- AJAX
  - `load()`
    - $(selector).load(URL,data,callback);
    - 示例：$("#div1").load("demo_test.txt #p1");
  - `$.get`