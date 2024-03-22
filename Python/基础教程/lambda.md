

```python
filter(lambda x: x % 3 == 0, [1, 2, 3])

sorted([1, 2, 3, 4, 5, 6, 7, 8, 9], key=lambda x: abs(5-x))

map(lambda x: x+1, [1, 2,3])

reduce(lambda a, b: '{}, {}'.format(a, b), [1, 2, 3, 4, 5, 6, 7, 8, 9])
```




--------

参考资料：
- [谈谈 Python lambda 表达式](https://juejin.cn/post/7151210802865766437)