
`2023.03.21`

添加 google analytics


`2023.04.08`

docsify\@4.js 本地化，添加 footer


`2023.07.31`

(1) 删除网址导航，使用最简单的浏览器收藏夹算了

整个文档博客系统：（555个文件，6.32MB）-> （521个文件，3.58MB）

chrome: 工作收藏夹; edge: 日用收藏夹

(2) jsdeliver 挂掉了

cdn.jsdelivr.net/npm -> fastly.jsdelivr.net/npm


`2023.08.09`

> blockquote

?> p.warn

!> p.tip

（1）添加更多的文本样式:

<p class="pyellow">使用 p class = "pyellow" 来写文本吧</p>

<p class="ppurple">使用 p class = "ppurple" 来写文本吧</p>

<p class="pgreen">使用 p class = "pgreen" 来写文本吧</p>

（2）将 https://cdn.bootcdn.net/ajax/libs/docsify/4.13.0/themes/vue.min.css 本地化，并集成 my.css

便于修改，修改了:
- 删除 p.tip:before 并修改 p.tip
- 删除 code.token
- 删除字体引入


`2023.08.25`

增加目录划线，美化目录条目过多的情况。






</br></br>

------------

字数统计、搜索插件 [flexsearch](https://github.com/nextapps-de/flexsearch) 不弄了