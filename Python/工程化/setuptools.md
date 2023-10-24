
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