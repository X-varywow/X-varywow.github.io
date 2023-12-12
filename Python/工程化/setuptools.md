
## _setuptools_

Python标准的打包及分发工具，用来替代 distutils



```python
# setup.py

# from setuptools import setup
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
```

```bash
# build_ext, 给 python 编译一个 c/c++ 扩展
# --inplace, 编译后的扩展放到原目录
python setup.py build_ext --inplace 
```

```bash
# 安装到当前环境的 site-pakages 下
python setup.py install 
```

```bash
python install .
```


.pyx 文件是由Cython编程语言编写的python扩展模块源代码文件。

需要先编译成 c 文件，之后再编译成.so文件(Linux平台)或.pyd文件(Windows平台)，才能作为模块 import 导入使用。


-------

参考 http://github.com/sublee/trueskill 中 steup 使用

```python
# include __about__.py
# about 中是这样格式：__version__ = '0.4.5'


__dir__ = os.path.dirname(__file__)
about = {}
with open(os.path.join(__dir__, 'trueskill', '__about__.py')) as f:
    exec(f.read(), about)

setup(
    name='trueskill',
    version=about['__version__'],
    license=about['__license__'],
    author=about['__author__'],
    ...
)
```


exec(f.read(), about) 这种用法，6





--------

参考资料：
- [Python打包分发工具setuptools简介](http://www.bjhee.com/setuptools.html)
- [Distutils 模块介绍](https://docs.python.org/zh-cn/3/distutils/introduction.html)
- [Python 库打包分发(setup.py 编写)简易指南](https://blog.konghy.cn/2018/04/29/setup-dot-py/)



</br>

## _pyproject.toml_

pyproject.toml 是一个配置文件，被用来替代原有的 setup.py, requirements.txt, setup.cfg


1. **项目元数据**: 用来定义项目的名称、版本、作者等信息。

```toml
[project]
name = "example-project"
version = "0.1.0"
description = "An example project"
authors = [
    { name = "Author Name", email = "author@example.com" },
]
```

2. **依赖管理**: 明确列出项目所需的依赖。

```toml
[project]
...
dependencies = [
    "requests >= 2.25.1",
    "numpy"
]
```

3. **开发依赖**: 在 `[build-system]` 部分列出为了构建项目 (例如编译扩展) 而需要的依赖项。

```toml
[build-system]
requires = ["setuptools >= 40.6.0", "wheel"]
build-backend = "setuptools.build_meta"
```

4. **工具配置**: 可以用来为第三方工具，比如 `flake8`, `black`, `isort` 等设置配置参数。

```toml
[tool.black]
line-length = 88
include = '\\.pyi?$'
exclude = '''
/(
    \\.eggs
)/
```


demo:

```python
[build-system]
requires = ["setuptools", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
authors = [
    {name = "Josiah Carberry", email = "josiah_carberry@brown.edu"},
]
description = "My package description"
readme = "README.rst"
requires-python = ">=3.7"
keywords = ["one", "two"]
license = {text = "BSD-3-Clause"}
classifiers = [
    "Framework :: Django",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "requests",
    'importlib-metadata; python_version<"3.8"',
]
dynamic = ["version"]

[project.optional-dependencies]
pdf = ["ReportLab>=1.2", "RXP"]
rest = ["docutils>=0.3", "pack ==1.1, ==1.3"]

[project.scripts]
my-script = "my_package.module:function"
```


---------

参考资料：
- [Python 新规范 pyproject.toml 完全解析](https://cloud.tencent.com/developer/article/2219745)
- https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html