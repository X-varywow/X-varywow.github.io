
想用 fastapi 的 html response, 但是部署到 k8s 就各种问题，，

想用 nodejs 新起一个前端服务，但是没有复用的路，估计较长的时间，，

想用 gradio, 但 mac 报 no Blocks ， 奇怪，

用 streamlit 了， 刚好


## preface

[cheats-sheet](https://cheat-sheet.streamlit.app/)

[create a multipage app](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app)

[30days](https://30days.streamlit.app/)


```python
# pip install streamlit

import streamlit as st

st.markdown("## hello world")

# streamlit run app.py
```

其它启动方式:

```python
import subprocess

# 在后台启动 Streamlit 服务
def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py", "--server.baseUrlPath=front"])
```

## config

查看 config 文档

```bash
streamlit config show
```

```bash
vim ~/.streamlit/config.toml
```

## pandas 支持

```python
df = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40]
     })
st.write(df)
```

连接 snowpark

```python
st.experimental_connection('pets_db', type='sql')
conn = st.experimental_connection('sql')
conn = st.experimental_connection('snowpark')

class MyConnection(ExperimentalBaseConnection[myconn.MyConnection]):
    def _connect(self, **kwargs) -> MyConnection:
        return myconn.connect(**self._secrets, **kwargs)
    def query(self, query):
       return self._instance.query(query)
```

## other

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



