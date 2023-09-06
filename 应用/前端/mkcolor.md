
?>  2022.03.26：找不到一个合适的配色网站；尤其是莫兰迪，自己动手做一个。

参考网站
- [CSDN-莫兰迪色系表](https://blog.csdn.net/weixin_44368437/article/details/114933796)
- https://colordrop.io/
- https://colorhunt.co/
- [中国配色网站](https://colors.ichuantong.cn/)

打算实现方式：
- html + css + js + json
- json 用于存放颜色的数据
- float 布局就可以了
- DOM 结构
  - head-wrap
    - h1
    - card-wrap
      - card-color
      - card-info

## 1. html

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/png" href="/main/img/favicon.png">
    <link rel="stylesheet" href="/main/css/color.css">
    <script src="/main/js/jquery-1.11.1.min.js"></script>
    <script src="/main/js/color.js"></script>
    <title>配色板</title>
</head>

<body>
    <div class="container">
        <div class="head-wrap">
            <h1>莫兰迪</h1>

            <div class="card-wrap">
                <div class="card-color" style="background:#e0e5df"></div>
                <div class="card-info">
                    <span class="hex" onclick="copyContent(this)">#e0e5df</span>
                </div>
            </div>

        </div>
    </div>

    <input id="copy_content" type="text" value="" style="position: absolute;top: 0;left: 0;opacity: 0;z-index: -10;" />
</body>
</html>
```

> html 只是一个例子，后面的网页实际是 javascript 读取 json 生成的


## 2. css

```css

body{
    font-family: monospace;
}
h1{
    text-align: center;
    float: left;
    width: 20%;
    font-family: Simsun;
    margin-left: 40%;
    margin-right: 40%;
    background-color: #B5C4B1;
    padding: 10px;
    border-radius: 5px;
}
.container{
    width: 94%;
    margin-left: 3%;
    margin-right: 3%;
}

.card-wrap{
    float: left;
    margin-left: 30px;
    text-align: center;
    margin-bottom: 30px;
}

.card-color{
    width: 200px;
    height: 100px;
    border-radius: 10px;
}

.card-info{
    margin-top: 0.5rem;
}

.hex{
    font-size: 0.8rem;
    padding: 0.3rem;
    background: #f2f2f2;
    border-radius: 0.2rem;
    text-transform: uppercase;
    cursor: pointer;
}
```

## 3. javascript

```js
// 利用 input 实现复制功能
function copyContent(ElementObj) {
    //var clickContent = grbToHex(ElementObj.style.background);
    var clickContent = ElementObj.id;
    console.log("已复制：", ElementObj.id)
    var inputElement = document.getElementById("copy_content");
    inputElement.value = clickContent;
    inputElement.select();
    document.execCommand("Copy");
}

// json 写入 html
// 网页中 $ 前的 \ 不用加，这里博客js渲染与 katex 冲突出错了我才加的
function jsonTohtml(d) {
    //console.log(d);
    d.forEach(e => {    //json -> html
        c2 = ""
        e.children.forEach(colorinfo => {
            c2 += `<div class="card-wrap">
                <div class="card-color" onclick="copyContent(this)" id="\${colorinfo.color}" style="background:\${colorinfo.color}"></div>
                <div class="card-info">
                    <span class="hex">\${colorinfo.info}</span>
                </div>
                </div>` });
        c1 = `<h1 id="\${e.head}" style="background:\${e.children[2].color}">\${e.head}</h1>`

        card = `<div class="head-wrap">${c1}${c2}</div>`
        $(".container").append(card);

        side_str = `<a href="#\${e.head}">
            <div class="btn-wrap" style="background:\${e.children[2].color}">
                <p>\${e.head}</p>
            </div>
        </a>`
        $(".side-wrap").append(side_str);
    });
}

// jquery
$.getJSON("color.json", function (d) {
    jsonTohtml(d);
})
```

## 4. json 

```json
[   
    {
        "head": "中国颜色",
        "children":[
            {
                "color":"#f2be45",
                "info":"赤金"
            }
        ]
    }
]
```

## 5. python

自动生成 json 文本的脚本（或使用 python 中的json模块）：

```python
import re

# 这里省略了数据
raw = """<tr><td bgcolor="#c1cbd7"></td><td>#c1cbd7</td><td>(193, 203, 215)</td><td bgcolor="#c1cbd7"></td></tr>"""

findColor = re.compile(r'bgcolor="(.*?)"')
data = re.findall(findColor, raw)

for c in data[::2]:
    print(""" {"color":" """, end="")
    print(c, end="")
    print(""" ","info":" """, end="")
    print(c, end="")
    print(""" "}, """, end="\n")
```

其它网站相同思路：复制 html 文本，正则匹配出标题、颜色。






