

垃圾回收是一种动态存储管理技术，gc 是 python 垃圾回收器接口。


垃圾回收的主要任务就是确定哪些对象可以被收回，并选择一个最佳的时间来释放它们占用的内存。标准的 CPython GC 有两个组件：
- 引用计数收集器(reference counting collector)：主要的、基础模块、不可控，不能禁用。
- 分代垃圾收集器(generational garbage collector)：辅助的、可以控制，即 gc module。


```python
# 启用自动垃圾回收
gc.enable()

gc.disable()

# 若被调用时不包含参数，则启动完全的垃圾回收。可选的参数 generation 可以是一个整数，指明需要回收哪一代（从 0 到 2 ）的垃圾。
gc.collect(generation=2)
```

pandas 被删除的列可能站着内存空间，

```python
for col in [
    'MAX_SCORE',
    'MAX_DAUB_SCORE',
    'MAX_BONUS']:
    del data[col]

gc.collect()
```


-----------

内存不够用时，linux 内核会通过三种机制来处理：内存回收、内存规整，oom-kill

内存回收时涉及写磁盘，导致 IO 激增


-----------

参考：
- [gc 官方文档](https://docs.python.org/zh-cn/3/library/gc.html)
- [GC 机制探究之 Python 篇](https://zhuanlan.zhihu.com/p/295062531)
- [详细的底层原理：Linux内存变低会发生什么问题](https://mp.weixin.qq.com/s/c2y36IH-4mRwhR-xvvdqGw)