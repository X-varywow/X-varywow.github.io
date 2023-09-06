
pathlib -- 面向对象的文件路径


```python
from pathlib import Path

p = Path(".")

# 列出子目录
[x for x in p.iterdir() if x.is_dir()]

# 列出当前目录树下所有 py 文件
list(p.glob('**/*.py'))

# 路径拼接
p = p / 'init.d' / 'reboot'


# 路径组成
p.name
p.stem # 不含后缀的文件名
p.suffix
p.parent


# 其他操作
Path.cwd()
Path.home()
str(Path.cwd())

```

参考资料：
- [官方文档](https://docs.python.org/zh-cn/3/library/pathlib.html)
- [python路径操作新标准：pathlib 模块](https://zhuanlan.zhihu.com/p/139783331)
- [pathlib介绍-比os.path更好的路径处理方式](https://zhuanlan.zhihu.com/p/33524938)