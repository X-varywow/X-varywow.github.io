该模块提供了一些变量和函数。这些变量可能被解释器使用，也可能由解释器提供。这些函数会影响解释器。本模块总是可用的。


`sys.path`

一个由字符串组成的列表，用于指定模块的搜索路径。
程序启动时将初始化该列表，列表的第一项 path[0] 目录含有调用 Python 解释器的脚本。如果脚本目录不可用（比如以交互方式调用了解释器，或脚本是从标准输入中读取的），则 path[0] 为空字符串，这将导致 Python 优先搜索当前目录中的模块

`sys.argv`

一个列表，其中包含了被传递给 Python 脚本的命令行参数。 argv[0] 为脚本的名称（是否是完整的路径名取决于操作系统）。如果是通过 Python 解释器的命令行参数 -c 来执行的， argv[0] 会被设置成字符串 '-c' 。如果没有脚本名被传递给 Python 解释器， argv[0] 为空字符串。

`sys.setrecursionlimit(limit)`

Set the maximum depth of the Python interpreter stack to limit. This limit prevents infinite recursion from causing an overflow of the C stack and crashing Python.