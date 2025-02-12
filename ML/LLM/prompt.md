
https://api-docs.deepseek.com/zh-cn/prompt-library/


三个基本角色：

| 角色      | 作用                      |
| --------- | ------------------------- |
| system    | 设定 AI 基础行为准则      |
| user      | 用户提问/追加提问         |
| assistant | ai 的历史回复，保持连贯性 |




### demo. 自动编写提示词

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="<YOUR_API_KEY>"
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
                "role": "system",
                "content": "你是一位大模型提示词生成专家，请根据用户的需求编写一个智能助手的提示词，来指导大模型进行内容生成，要求：\n1. 以 Markdown 格式输出\n2. 贴合用户需求，描述智能助手的定位、能力、知识储备\n3. 提示词应清晰、精确、易于理解，在保持质量的同时，尽可能简洁\n4. 只输出提示词，不要输出多余解释"
        },
        {
                "role": "user",
                "content": "请帮我生成一个“Linux 助手”的提示词"
        }
    ]
)

print(completion.choices[0].message.content)
```

输出：

```md
# Linux 助手提示词

## 定位
- **Linux 助手**：一个专注于提供 Linux 操作系统相关知识和支持的智能助手。

## 能力
- **系统管理**：能够提供 Linux 系统安装、配置、维护和优化的指导。
- **命令行操作**：熟练掌握并解释各种 Linux 命令行工具和脚本。
- **故障排除**：帮助用户诊断和解决常见的 Linux 系统问题。
- **软件管理**：指导用户进行软件包的安装、更新和卸载。
- **安全性**：提供 Linux 系统安全设置和最佳实践的建议。

## 知识储备
- **操作系统**：深入了解各种 Linux 发行版，如 Ubuntu、CentOS、Debian 等。
- **网络配置**：熟悉 Linux 网络设置和配置，包括防火墙、路由和网络服务。
- **文件系统**：掌握 Linux 文件系统结构和管理。
- **编程环境**：了解在 Linux 上搭建和使用各种编程环境，如 Python、Java 等。

## 使用示例
- **用户问题**：如何安装 Apache 服务器？
- **助手响应**：可以使用以下命令安装 Apache 服务器：

  sudo apt update
  sudo apt install apache2

```


### demo. 格式化输出

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="<YOUR_API_KEY>"
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
                "role": "system",
                "content": """
用户将提供给你一段新闻内容，请你分析新闻内容，并提取其中的关键信息，以 JSON 的形式输出，输出的 JSON 需遵守以下的格式：
{
  "entiry": <新闻实体>,
  "time": <新闻时间，格式为 YYYY-mm-dd HH:MM:SS，没有请填 null>,
  "summary": <新闻内容总结>
}
"""
        },
        {
                "role": "user",
                "content": "8月31日，一枚猎鹰9号运载火箭于美国东部时间凌晨3时43分从美国佛罗里达州卡纳维拉尔角发射升空，将21颗星链卫星（Starlink）送入轨道。紧接着，在当天美国东部时间凌晨4时48分，另一枚猎鹰9号运载火箭从美国加利福尼亚州范登堡太空基地发射升空，同样将21颗星链卫星成功送入轨道。两次发射间隔65分钟创猎鹰9号运载火箭最短发射间隔纪录。\n\n美国联邦航空管理局于8月30日表示，尽管对太空探索技术公司的调查仍在进行，但已允许其猎鹰9号运载火箭恢复发射。目前，双方并未透露8月28日助推器着陆失败事故的详细信息。尽管发射已恢复，但原计划进行五天太空活动的“北极星黎明”（Polaris Dawn）任务却被推迟。美国太空探索技术公司为该任务正在积极筹备，等待美国联邦航空管理局的最终批准后尽快进行发射。"
        }
    ]
)

print(completion.choices[0].message.content)
```

输出

```json
{
  "entity": "猎鹰9号运载火箭",
  "time": "2023-08-31 03:43:00",
  "summary": "8月31日，猎鹰9号运载火箭两次成功发射，将42颗星链卫星送入轨道，创下了最短发射间隔纪录。尽管美国联邦航空管理局允许恢复发射，但原计划的“北极星黎明”任务被推迟，等待最终批准。"
}
```