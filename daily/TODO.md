
生成式 AI,确实使游戏开发快很多；https://huggingface.co/cagliostrolab/animagine-xl-3.0

语音、图片、线稿生图

https://instantid.github.io/


思考一下要做的, 学点 RL ，感觉没啥好开发的

https://github.com/KRTirtho/spotube


秋辞长离去，辗转何时归。京花入山门，落我长相思。

是该写点东西了，

兰亭集序，赤壁赋，滕王阁序


弄个自动化控制，强化学习工具，先把界面弄出来

POE 角色扮演，确实有趣

https://docs.python.org/zh-cn/3/library/unittest.mock.html

https://testdriven.io/blog/developing-a-single-page-app-with-fastapi-and-vuejs/

https://testdriven.io/blog/fastapi-crud/


https://mp.weixin.qq.com/s/SZOwdrOd2CAZLrU5A12cnA

https://huggingface.co/spaces/XzJosh/LAPLACE-Bert-VITS2-2.3




https://www.nature.com/

GO VUE 前后端联动

https://lilianweng.github.io/


https://github.com/valinet/ExplorerPatcher/wiki/All-features


整理 python/数据库 ./ml 面试等，补充理论即可，缺失重要


- [ ] 课程笔记整理
- [ ] 更新 develop
- [ ] 系统论的书
- [ ] MHA
- [ ] chatglm3
- [ ] vgg 代码写法，看起来  vits 代码不够优雅
- [ ] 多读 torch 文档


https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master


holocubic, 作用：显示时钟，图片，温度信息

ML/DL/ROADMAP

huawei ai tutorial

https://github.com/ZachGoldberg/Startup-CTO-Handbook/blob/main/StartupCTOHandbook.md#speed-is-your-friend

- [ ] blog ML/DL/transformer 整理

[深度学习之图像翻译与风格化GAN-理论与实践](https://www.bilibili.com/video/BV1Wr4y1b77B)


- electron开发，代替 TODO，自定义背景，等


- [ ] 右侧添加一个单页面的 目录
- [ ] 整理 torch 文档到 blog


前端应用程序开发：https://github.com/Moonvy/OpenPromptStudio

https://github.com/wangshub


https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3

http://nlp.seas.harvard.edu/2018/04/03/attention.html

https://www.cloudskillsboost.google/journeys/118


https://github.com/MichaelCade/90DaysOfDevOps/blob/main/2023/day06.md

https://openai.com/sora 质量有点好，但是再长一些的连贯，制作短片，将文本内容转视频，辅助视频制作，会很有用

https://mp.weixin.qq.com/s/WXoSnUXjrn_6EzSM1MQPTw

vae diffusion transformer 数理基础，算子，有空补一下;


----------
有意思，
https://github.com/BAAI-Agents/Cradle

llm agent, 基于大型语言模型（如GPT系列）的智能体或助手

https://github.com/a-real-ai/pywinassistant


使用 ahk 做输入控制

图意多模态理解，（text: ocr, 图意？？）

把按键控制定义 skill

- atomic_skills
- composite_skills


决策，控制

层层抽象，

cv_go_to_icon screenshot -> match(mtm, cv2.TM_CCOEFF_NORMED)

https://github.com/multi-template-matching/mtm-python-oop

pyautogui.getWindowsWithTitle(config.env_name)

重点在那一大片提示词, 然后 gpt 理解 图像

关键点： client 学习与决策？图意理解？

https://github.com/nashsu/FreeAskInternet

https://github.com/PKU-YuanGroup/Open-Sora-Plan

trafilatura

simcse

https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web

https://m.ofweek.com/ai/2024-01/ART-201712-8440-30623318.html gpt3.5 弄个 agent


IC-Light sd 光照工具

------------

https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/

llm, sd; blog, python-proj

https://github.com/liguodongiot/llm-action


https://sci-hub.ru/https://www.sciencedirect.com/science/article/abs/pii/030646039390038B

sci-hub, 6的； ocr

https://mp.weixin.qq.com/s/6Jn4-3KPoffsYGrrvYX6vg



wx or unity 类小丑牌小程序开发


https://web.archive.org/web/20181103114010/https://mlwave.com/kaggle-ensembling-guide/

0517
- [ ] 清旧文件
- [ ] 100 lv
- [ ] lf

`<u>`

-----------

1 个 Token 大约相当于 1.5-2 个汉字

kimi 真的可以，介绍并总结这个网页上的内容：https://mp.weixin.qq.com/s/WXoSnUXjrn_6EzSM1MQPTw

直接就出来了，chatgpt 好像还不行。

poe、chatgpt api 需要订阅; kimi 半免费


[moonshot console](https://platform.moonshot.cn/console/) 新建 api

如下调用即可

```python
# pip install openai
from openai import OpenAI
 
client = OpenAI(
    api_key = "。。。",
    base_url = "https://api.moonshot.cn/v1",
)
 
completion = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        # {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
        {"role": "user", "content": "介绍并总结这个网页上的内容：https://mp.weixin.qq.com/s/6Jn4-3KPoffsYGrrvYX6vg"}
    ],
    temperature = 0.3,
)
 
print(completion.choices[0].message.content)
```

api 服务不如网页服务

curl https://api.moonshot.cn/v1/users/me/balance -H "Authorization: Bearer .."

更多： langchain 工作流、界面/机器人


----------------

暗黑4

优点：
- 太古词缀，回火、大小秘境、创新性；
- 为玩家减负，威能获取；
- 怪物密度、成长曲线能吸引玩家；
- 物品稀有度、掉率控制得可以，也不必氪金。

缺点：
- 职业强度太依赖回火词条，就野蛮人基本只能玩两个流派，设计师教你玩游戏
- 排行榜、好友界面、地图 bug
- 大后期体验一般、基本30h 之后，不新增活动的话，只剩下冲层了；（每周应该弄个独特活动，或者长期的联盟、周本）