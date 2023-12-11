
```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/example', methods=['GET'])
def example_get():
    # 处理 GET 请求的逻辑
    return '处理了GET请求'

@app.route('/example', methods=['POST'])
def example_post():
    # 处理 POST 请求的逻辑
    return '处理了POST请求'
```


感觉，fastapi 好用一些（用的较少），自动的接口文档，pydantic 等



---------------

参考资料：
- [官方文档](https://flask.palletsprojects.com/en/3.0.x/quickstart/)