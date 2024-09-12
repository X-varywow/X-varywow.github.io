

## _unittest_

`unittest`, 利用多种断言方法，来定义测试用例，对代码进行测试。


</br>

demo1:

```python
import unittest

class case1(unittest.TestCase):

    # setUp() 会在每个测试用例之前执行
    def setUp(selef) -> None:
        self.arr = [1]       # 后续每个 test 都能使用 self.arr 来访问
    
    # tearDown() 会在每个测试用例之后执行
    def tearDown(self):
        pass


    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_in(self):
        self.assertIn("a", "ab")

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()

```

</br>


```bash
export PYTHONPATH=$PYTHONPATH:/Users/yourname/yourpath
```



</br>

## _unittest.mock_

允许使用模拟对象来替换受测系统的部分，并对这部分进行断言判断。

在隔离环境中测试单个组件，适用于真实依赖难以构建或设置。

mock 常用场景：
- 模拟函数调用
- 记录在对象上的方法调用

```python
# calculator.py
def add(x, y):
    return x + y


# test_calculator.py
import unittest
from unittest import TestCase
from unittest.mock import patch
import calculator

class TestCalculator(TestCase):
    @patch('calculator.add')
    def test_add(self, mock_add):
        # 设置 mock 对象的返回值
        mock_add.return_value = 10
        
        # 调用被测试的函数
        result = calculator.add(3, 4)
        
        # 断言函数返回了预期的结果
        self.assertEqual(result, 10)
        
        # 断言 add 函数被调用了一次
        mock_add.assert_called_once_with(3, 4)


if __name__ == '__main__':
    unittest.main()
```

这里 calculator.add 并不会调用，是将设置的 return_value 返回了



## 单元测试的意义

对于功能性函数，如（leetcode中的题目），单元测试可以保证这个函数的正确性和健壮性，无论内部怎样修改。（实际上，自己写的单测，一般都是已经想到的，加上单测并不能改变啥）

对于业务性函数，结果不具有明确的正确和错误之分，若还要拉取动态特征数据，很难。

对于有着明确正确错误，代码内部修改比较频繁，单测和线上容易保持一致的任务，单元测试是很有必要的



------------

参考资料：
- [unittest 官方文档](https://docs.python.org/zh-cn/3/library/unittest.html)
- https://docs.python.org/zh-cn/3/library/unittest.mock.html
- [Is Unit Testing worth the effort? ](https://stackoverflow.com/questions/67299/is-unit-testing-worth-the-effort)
- https://meik2333.com/posts/unit-testing-in-python/
