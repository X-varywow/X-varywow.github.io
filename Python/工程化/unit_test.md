

## _unittest_

`unittest`, 利用多种断言方法，来定义测试用例，对代码进行测试。


</br>

demo1:

```python
import unittest

class case1(unittest.TestCase):

    # setUp() 会在每个测试用例之前执行
    def setUp(selef) -> None:
        pass
    
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







------------

参考资料：
- [unittest 官方文档](https://docs.python.org/zh-cn/3/library/unittest.html)
- https://docs.python.org/zh-cn/3/library/unittest.mock.html
- [Is Unit Testing worth the effort? ](https://stackoverflow.com/questions/67299/is-unit-testing-worth-the-effort)
- https://meik2333.com/posts/unit-testing-in-python/
