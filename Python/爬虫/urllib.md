
```python
from urllib.request import urlopen 

html = urlopen("https://www.baidu.com/")

# 获取的html内容是字节，将其转化为字符串
html_text = bytes.decode(html.read())

print(html_text)
```

```python

from urllib.request import urlopen
# BeautifulSoup
from bs4 import BeautifulSoup as bf

html = urlopen("https://www.baidu.com/")
# 用BeautifulSoup解析html
obj = bf(html.read(),'html.parser')
# 从标签head、title里提取标题
title = obj.head.title
# 只提取logo图片的信息
logo_pic_info = obj.find_all('img',class_="index-logo-src")
# 提取logo图片的链接
logo_url = "https:"+logo_pic_info[0]['src']

print(logo_url)
```