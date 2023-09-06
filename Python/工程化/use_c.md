
Python 是一种动态类型语言，在运行时才去做数据类型检查，也就是说 **动态解释的过程导致了效率较低**；

为了给 python 加速，常常在代码中结合 C/C++；

## _pybind11_

简化 Python 调用 C++ 代码的库。这是一个仅头文件的 C++ 库，它可以将 C++ 代码转化成 Python 可直接引用的模块，轻松实现 Python 调用 C++  代码。通过这种混合编程的方式，可以提高 Python 代码的性能。


```bash
#手动编译 C++ 代码
$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

#然后在 Python 代码中直接 import 即可使用
$ python
Python 3.9.10 (main, Jan 15 2022, 11:48:04)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import example
>>> example.add(1, 2)
3

```



----------

参考资料：
- [Python调用C/C++: cython及pybind11](https://zhuanlan.zhihu.com/p/442935082)
- [pybind11 官方文档](https://pybind11.readthedocs.io/en/stable/)