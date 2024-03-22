_preface_

这个模块提供了堆队列算法的实现，也称为 **优先队列** 算法。

堆是一个二叉树，它的每个父节点的值都只会小于或等于所有孩子节点（的值）。 它使用了数组来实现：从零开始计数，对于所有的 k ，都有 `heap[k] <= heap[2*k+1]` 和 `heap[k] <= heap[2*k+2]`。 为了便于比较，不存在的元素被认为是无限大。 堆最有趣的特性在于最小的元素总是在根结点：heap[0]。


Pyhton 中默认为 **最小堆**，pop出来的是最小的`heap[0]`

参考资料：[官方heapq文档](https://docs.python.org/zh-cn/3/library/heapq.html?highlight=heapq#module-heapq)


### heapify(x)

将`list x` 转换成堆，原地，**线性时间**内。


### heappush(heap, item)
将 `item` 的值加入 `heap` 中，保持堆的不变性。


### heappop(heap)
弹出并返回 `heap` 的最小的元素，保持堆的不变性。

### heappushpop(heap, item)
将 `item` 放入堆中，然后弹出并返回 `heap` 的最小元素。


<br><br>

>注意列表转换为堆 是 线性时间，
>所以在一些算法问题的求解中，不使用排序而使用堆，会使效率更高：






