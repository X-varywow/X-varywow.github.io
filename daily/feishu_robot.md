
python  使用方式：

```python
import requests
import json

## 替换为你的自定义机器人的 webhook 地址。
url = "[http](https://open.feishu.cn/open-apis/bot/v2/hook/,,,)"

## 将消息卡片内容粘贴至此处。
card = json.dumps({
    "elements":[
        {

        },
        {

        }
    ]
})



body =json.dumps({"msg_type": "interactive","card":card})
headers = {"Content-Type":"application/json"}

res = requests.post(url=url, data=body, headers=headers)

print(res.text)
```



## 图片消息

https://open.feishu.cn/document/server-docs/im-v1/image/create

## MD 支持

https://open.feishu.cn/document/common-capabilities/message-card/message-cards-content/using-markdown-tags

查看 template 颜色




多列布局：
https://open.feishu.cn/document/common-capabilities/message-card/message-cards-content/column-set


BUTTON primary, default

文字颜色：green red grey default

标签：<text_tag color='red'>标签文本</text_tag>


## card

总体：
```python
card = json.dumps({
    "elements": [],
    "header": {},
    "config": {
        "wide_screen_mode": True
    }
})
```

组件：

```python
btn1 = {
    "tag": "button",
    
}

LINK_INFO = {
    "tags": "action",
    "actions": [btn1, btn2]
}
```



at 指定人员 https://www.cnblogs.com/mxcl/p/16359730.html


在 card 中使用方法：（以组件化加入到 card/elements 中）

```python
AT_SOMEONE = {
  "tag": "markdown",
  "content": "<at id=open_id>h</at>"
}
```



## _多列布局_

https://open.feishu.cn/document/common-capabilities/message-card/message-cards-content/column-set





## 其它说明

"background_style" red 就不行，使用内置的 grey

<font color='#f66'> 不支持自定义，只支持 red green 

DEBUG 模式，右键头像，可以获取 user_id


---------------

显示折叠消息：

官方文档上暂时没有这个组件，使用的如下方式：

```python
MORE_INFO = {
    "tag": "action",
    "actions":[{
        "tag": "select_static",
        "placeholder": {
        "tag": "plain_text",
        "content": "more info"
        },
        "options": [
        {
            "text": {
            "tag": "plain_text",
            "content": i
            },
            "value": ""
        }
        for i in params['more_info']
        ]
    }]
    }
```

