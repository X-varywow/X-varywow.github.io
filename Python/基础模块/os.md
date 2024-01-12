
!>本模块提供了一种使用与操作系统相关的功能的便捷式途径。

`os.path.abspath()`

`os.path.dirname()`

`os.path.exists()`

`os.path.getsize()`

`os.path.getatime()`

`os.path.isfile()` `os.path.isdir()`

`os.path.join()`


`os.getcwd()`

返回表示当前工作目录的字符串。

`os.getuid()`

`os.get_terminal_size()`

`os.listdir()`

`os.mkdir()`

`os.remove()`

`os.rename()`

`os.walk()`

```python
for root, dirs, files  in os.walk(path):
    print(root, dirs, files)
```


```python
# os 并不会自动展开 ~ 到用户的主目录...
os.path.exists("~/SageMaker/svc/pretrain/vits/pretrained_ljs.pth")
```





--------

参考资料：
- [官方文档 os.path](https://docs.python.org/zh-cn/3/library/os.path.html)
- [官方文档 os](https://docs.python.org/zh-cn/3/library/os.html)