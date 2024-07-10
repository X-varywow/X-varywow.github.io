

https://yanbin.blog/python-multi-envs-configurations/

python 自动管理 config ，根据环境不同起服务的工具


-------------

toml

- config.production.toml
- config.test.toml


```toml
[database]
server = "192.168.1.1"
ports = [ 8001, 8001, 8002 ]
connection_max = 5000
enabled = true

[server]
port = 8080
host = "localhost"
```



```python
import os
import toml

# 根据环境变量选择配置文件
env = os.getenv('ENV', 'test')  # 默认为测试环境
config_path = f'config.{env}.toml'

with open(config_path) as f:
    config = toml.load(f)

# 访问配置项
database_config = config['database']
server_config = config['server']

print(f"Database Server: {database_config['server']}")
print(f"Server Port: {server_config['port']}")
```

对于不能明文表示的密码等，可以再加一层转义，toml 中只记录对应的获取地址，然后 config.py 进行获取，其它所有文件读取配置只从 config.py 中读走