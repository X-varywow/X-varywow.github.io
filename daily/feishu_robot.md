
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


多列布局：
https://open.feishu.cn/document/common-capabilities/message-card/message-cards-content/column-set


BUTTON primary, default


## 消息体说明

总体：
```python
card = json.dumps({
    "elements": []
    "header": {}
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






## 其它说明

"background_style" red 就不行，使用内置的 grey

<font color='#f66'> 不支持自定义，只支持 red green 





-------


实例：python 定时任务，从 sf 拉数据，制作监控