
demo1. 加载模块&使用 getattr 获取入口函数并运行

```python
import importlib
runner_module = importlib.import_module(f'cradle.runner.{runner_key}_runner')
entry = getattr(runner_module, 'entry')
entry()
```

demo2. 动态加载配置文件

```python
config_dict = {}

def build_config():
    global config_dict
    
    config = importlib.import_module("config.constant")
    # update and append; not del key in dict
    config = importlib.reload(config)
    # 注意这里对字典重新赋值， 会引用新对象的内存地址; 使用 update 即可避免
    #（会影响到 本文件顶层的 config_dict; 不会影响到其它文件从这里引用的 config_dict，其它文件可通过read_config引入）
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}

def read_config(key):
    global config_dict
    return config_dict[key]
```

路径可通过如下方式修改：

```python
import sys
sys.path.append("..") 
sys.path
```
