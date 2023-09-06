
```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
```




---------------

参考资料：
- [官方文档](https://flask.palletsprojects.com/en/2.2.x/quickstart/)