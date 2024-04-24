## preface

```python
# pip install streamlit

import streamlit as st

st.markdown("## hello world")
```


启动方式:

(1)
```bash
streamlit run app.py
```
(2)
```python
import subprocess

# 在后台启动 Streamlit 服务
def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py", "--server.baseUrlPath=front"])
```


## 常用组件

使用 session_state 在会话中存储全局变量

https://docs.streamlit.io/library/api-reference/session-state

```python
st.text_input("Your name", key="name")

# This exists now:
st.session_state.name
```

```python
# 添加占位符
placeholder = st.empty()
# 创建进度条
bar = st.progress(0)

for i in range(100):
    time.sleep(0.05)
    # 不断更新占位符的内容
    placeholder.text(f"Iteration {i+1}")
    # 不断更新进度条
    bar.progress(i + 1)

# 状态
st.success("Finished")
```

```python
with st.sidebar:
    st.header("Configuration")
    day = st.number_input("距离当前时间天数", value = 3)

    if st.button("刷新数据"):
        # statement = refresh_qa_sql.format(day=day)
        # session.sql(statement)
        session.call("procedure", 1)
        st.success("success")
```

```python
def color_cells(val):
    if val > 90:
        color = '#ead0d1'
    elif val < 70:
        color = '#2add9c50' 
    else:
        color = '#e9e7ef50'
    return f'background-color: {color}'

styled_df = df.style.applymap(color_cells, subset=['col2'])

# st.write(df)
st.dataframe(styled_df)
```

```python
st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

st.code("""
[theme]
primaryColor="#F39C12"
backgroundColor="#2E86C1"
secondaryBackgroundColor="#AED6F1"
textColor="#FFFFFF"
font="monospace"
""")
```

更多组件，请查看参考资料（1）（2）



## config

查看 config 文档
```bash
streamlit config show

vim ~/.streamlit/config.toml
```
pageconfig:

```python
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide"
)
```


## Other

### 使用装饰器做权限验证和分级

st.session_state 相当于一个全局变量字典

```python
from functools import wraps

def md5_decorator(level):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if st.session_state.auth_level >= level:
                return func(*args, **kwargs)
            else:
                return "Please confirm your authoration!", 0
        return wrapper
    return decorator
    
    
@md5_decorator(level = 3)
def func(citus_connection_config, content):
    pass
```

### snowflake session

Snowflake 内使用：
```python
from snowflake.snowpark.context import get_active_session
session = get_active_session()
```
POD 中使用：

```python
from snowflake.snowpark import Session
session = Session.builder.configs(snowflake_config).create()
```

### Html 支持

```python
st.markdown('<br>', unsafe_allow_html=True)
```

------------

参考资料:

- [cheats-sheet](https://cheat-sheet.streamlit.app/)
- https://zhuanlan.zhihu.com/p/163927661
- [create a multipage app](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app)
- [30days](https://30days.streamlit.app/)
- Gallery: https://streamlit.io/gallery
- 布局方法：https://docs.streamlit.io/develop/api-reference/layout