
BeautifulSoup 是一个 html 解析器，根据标签快速获取有效信息。

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")
        # 找到所要信息所在的网页中的位置
    for item in soup.find_all('div', class_="item"):
        pass
```

[官方文档](https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/) [教程1](https://www.cnblogs.com/chenyangqit/p/16594745.html)