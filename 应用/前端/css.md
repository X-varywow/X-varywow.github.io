## 1. 样式的引入

**外部样式**

```html
<head>
<link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
```

**内部样式**

```html
<head>
<style>
hr {color:sienna;}
p {margin-left:20px;}
body {background-image:url("images/back40.gif");}
</style>
</head>
```

**内联样式**

```html
<p style="color:sienna;margin-left:20px">这是一个段落。</p>
```

优先级：内联样式 > 内部样式 > 外部样式


## 2. 选择器


**id 选择器**，可以为标有特定 id 的 HTML 元素指定特定的样式。

```css
#para1
{
    text-align:center;
    color:red;
}
```

**class 选择器**，用于描述一组元素的样式

```css
.center {text-align:center;}

p.center {text-align:center;}
```

**后代选择器** 以 ` ` 空格隔开


## 3. 常用属性

各种属性：
- background
- test-align
- font-family
- font-style
- font-size
- a:hover
- a:visited
- 盒子模型
  - margin, 外边距
  - border
  - padding, 内边距
  - content
- max-height
- line-height
- display:none
- visibility:hidden
- display:block; 作为块元素，占用全部宽度，前后都是换行符
- position
  - absolute 回合其它元素重叠
  - static 遵循正常的文档流对象
  - fixed 相对于浏览器窗口时固定位置
- z-index 指定了一个元素的堆叠顺序
- overflow: 溢出处理
- float
- margin: auto; 水平方向上居中对齐
- opacity
- ！important 增加样式的权重
- [导航栏](https://www.runoob.com/css/css-navbar.html)
- [提示工具](https://www.runoob.com/css/css-tooltip.html)
- [响应式卡片](https://www.runoob.com/css/css-image-gallery.html)
- [网页布局](https://www.runoob.com/css/css-website-layout.html)


</br>

## _css3_

- border-radius
- box-shadow(h-shadow, v-shwdow, bluur, spread, color)
- border-image
- [使用自定义字体](https://www.runoob.com/css3/css3-fonts.html)
- transform
- rotate
- transition， 过渡
- keyframes, 动画
- [按钮](https://www.runoob.com/css3/css3-buttons.html)
- [弹性盒子](https://www.runoob.com/css3/css3-flexbox.html)


</br>

## _一些框架_

bootstrap

[Tailwind CSS](https://tailwindcss.com/)

