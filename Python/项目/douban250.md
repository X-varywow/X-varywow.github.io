

一直想做一个 豆瓣250电影 看了多少的东西，

## v1

### 1. 准备豆瓣数据
```python
# 爬虫代码改自：https://blog.csdn.net/Mu_yongheng/article/details/114048247

from bs4 import BeautifulSoup      # 网页解析，获取数据
import re        # 正则表达式，进行文字匹配
import urllib.request, urllib.error    # 指定URL，获取网页数据
import xlwt      # 进行excel操作
import sqlite3   # 进行SQLite数据库操作

# 影片链接
findLink = re.compile(r'<a href="(.*?)">')  # 创建正则表达式对象（规则）
# 图片
findImgSrc = re.compile(r'<img.*src="(.*?)"')
# 片名
findTitle = re.compile(r'<span class="title">(.*)</span>')
# 影片评分
findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
# 评价人数
findJudge = re.compile(r'<span>(\d*)人评价</span>')

def getData(baseurl):
    datalist = []
    for i in range(0, 10):
        n = str(i*25)   # 页数
        url = baseurl + n  # 每一页的网址
        html = askURL(url)  # 访问每一个网页的内容

        # 逐一解析数据
        soup = BeautifulSoup(html, "html.parser")
        # 找到所要信息所在的网页中的位置
        for item in soup.find_all('div', class_="item"):
            # print(item)
            data = []  # 保存一部电影的全部信息
            item = str(item)
            # 获取到影片的超链接
            link = re.findall(findLink, item)[0]   # 获取电影链接
            data.append(link)  # 添加电影链接
            imgSrc = re.findall(findImgSrc, item)[0]  # 获取图片链接
            data.append(imgSrc)
            title = re.findall(findTitle, item)  # 获取电影名称
            if(len(title) == 2):                 # 区分中文名称和外文名称
                data.append(title[0])
                otitle = title[1].replace("/", "")
                data.append(otitle)
            else:
                data.append(title[0])
                data.append(' ')  # 如果没有外文名称则用空格占位，防止混乱
            rating = re.findall(findRating, item)[0]   # 获取评分
            data.append(rating)
            judge = re.findall(findJudge, item)[0]   # 获取评价人数
            data.append(judge)
            
            datalist.append(data)  # 将一步电影的所有爬取信息存入datalist列表

    return datalist    # 返回所有电影信息

def askURL(url):
    head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"}

    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)

    return html

def saveData(datalist, savepath):
    workbook = xlwt.Workbook(encoding="utf-8")  # 创建workbook对象
    worksheet = workbook.add_sheet('sheet1')  # 创建工作表
    # 创建列名
    col = ("影片链接", "图片链接", "影片中文名称", "影片外文名称", "影片评分", "评价人数")
    for i in range(0, 6):  # 写入列名
        worksheet.write(0, i, col[i]) # 列名
    for i in range(0, 250):  # 写入电影信息
        print("第{}行".format(i+1))
        data = datalist[i]
        for j in range(0, 6):
            worksheet.write(i+1, j, data[j])
    # 保存文件
    workbook.save(savepath)

def main():
    # 要爬取的网址
    baseurl = "https://movie.douban.com/top250?start="
    # 1.爬取网页
    datalist = getData(baseurl)
    # 3.保存数据
    savepath = "豆瓣电影Top250.xls"
    saveData(datalist, savepath)

if __name__ == "__main__":
    # 调用函数
    main()
    print("爬取完毕！")
```

### 2. 准备自己的数据

对着豆瓣250的单子输入 `1` `0` `2`
- 1 表示 看过
- 0 表示 没看过
- 2 表示 不清楚

```python
res = "1111111211111111111111010111112101021001122101200021111212111111101111101010110001002000000010101101001100111000011101100010011100010002020100000000100000200100022201000011000001000000000000000000002000200000000010000000000000000200002000000010010021"

print(len(res))
```

后续，应该可以爬取查看自己看过没

### 3. 准备基础的前端

```html
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8"> 
        <title>豆瓣250打卡</title> 
        <style>
        .flex-container {
            display: -webkit-flex;
            display: flex;
            -webkit-justify-content: center;
            justify-content: center;
            width: 100%;
            background-color: #fefdca;
        }
        
        .flex-item {
            border-radius: 10px;
            background-color: #a5dee5;
            width: 8%;
            height: 100px;
            margin: 10px;
        }

        .introduce {
            padding-top: 2rem;
            height: 10rem;
            background-color: #fefdca;
        }

        .color1 {
            background-color: #e0f9b5;
        }

        .color0 {
            background-color: #ffcfdf;
        }

        .color2 {
            background-color: #a5dee5;
        }

        h1{
            text-align: center;
        }

        p{
            text-align: center;
        }

        .flex-item a{
            color: #000;
            font-weight: bold;
            font-size: 1.4rem;
            font-family: 'SimSun', Courier, monospace;
            text-decoration: none;
        }
        </style>
    </head>
<body>
    <div class = "introduce">
        <h1>公告</h1>
        <p>这是一个豆瓣250刷电影的页面。欢迎访问啊</p>
    </div>
    <div class="flex-container">
        <div class="flex-item color1">001<br><br><a href="https://movie.douban.com/subject/1292052/" target="_blank">《肖申克的救赎》<br></a></div>
    </div>   
</body>
</html>
```

### 4. 生成部分HTML

先查看第一步得到的数据：
```python
import pandas as pd
df = pd.read_excel("豆瓣电影Top250.xls")
df.head(10)
```

之后：
```python
for i in range(25):
    print("<div class=\"flex-container\">")
    for j in range(10):
        idx = i*10+j
        print("<div class=\"flex-item color{}\">{:03d}<br><br><a href=\"{}\" target=\"_blank\">《{}》<br></a></div>".format(res[idx],idx+1,df.loc[idx,"影片链接"],df.loc[idx,"影片中文名称"]))
    print("</div>")
```

再最后，将自动生成的代码复制到 HTML 的 64~66 行即可。


这是 [页面展示](../zone/250.html)

### 5. 后续


可以加入js使区块的颜色切换并保存，加入统计，。
加入响应式等，小屏体验不好


不过，应该没啥后续了，

>前端其实挺有趣的，看着自己一个盒子一个盒子敲出来，然后对着几个样例网站更改样式，不会的属性去搜一下。
>做完之后挺有成就感的。


## v2

2021-11-04

>改进：
>- 从 ipynb 改到 py
>- 加入进度条
>- json数据持久化
>- 命令行交互体验提升
>- tooltip 显示影片短评

#### 1. 创建虚拟环境

在项目文件夹下运行：

```cmd
py -m venv env
```

即可在该文件夹下生成一个名为 env 的虚拟环境，方便后续的 freeze 以及依赖管理

然后激活虚拟环境，并安装相关依赖：

```cmd
cd env/Scripts
activate

pip install pandas
pip install bs4
pip install tqdm
```


#### 2. 代码结构

[开源地址](https://github.com/X-varywow/vis_page_builder)

#### 3. 导出依赖

```cmd
pip freeze>requirements.txt
```

发布后使用者需进行的操作：

```cmd
pip install -r requirements.txt
```

## v2.1

元编程（Metaprogramming）是编写、操纵程序的程序，简而言之即为用代码生成代码。

[一文读懂元编程](https://www.jianshu.com/p/d3b637ece518)


需求分析：
- 主要是前端，做个怎么样的展示
- 爬取最新的豆瓣 Top250

试了试 bootstrap card，加图片展示效果并不好。

这个页面偏统计意义了，挺好的

转策略了，在已有代码改改

```python
# raw 是一个 html 文本
# 所有已有关于电影的信息
def gen_vis(raw):
    findTitle = re.compile(r'>《(.*)》')
    findColor = re.compile(r'color(\d)')
    vis = {}


    soup = BeautifulSoup(raw, "html.parser")

    for item in soup.find_all("div", class_="flex-item"):
        item = str(item)
        title = re.findall(findTitle, item)[0]
        color = re.findall(findColor, item)[0]
        vis[title] = color
        
    return vis

vis = gen_vis(raw)
```

```python
# df 是新爬取的，参考：旧版 v2 代码
def gen_new_html():
    loss = []
    for i,row in enumerate(df):
        title = row[2]
        if title not in vis:
            loss.append(row)
            
    return loss
```

```python
from math import ceil
loss = gen_new_html()
print(loss)

for i in range(ceil(len(loss)/10)):
    print("<div class=\"flex-container\">")
    for j in range(10):
        idx = i*10 + j
        if idx < len(loss):
            row = loss[idx]
            print("<div class=\"flex-item color{}\">{:03d}<a href=\"{}\" data-bs-toggle='tooltip' data-bs-placement='top' title='{}' target=\"_blank\">《{}》</a></div>".format(0,idx+1,row[0],row[3],row[2]))

    print("</div>")
```