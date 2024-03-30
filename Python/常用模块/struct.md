官方文档：https://docs.python.org/zh-cn/3/library/struct.html

将字节串解读为打包的二进制数据

`struct.pack(format, v1, v2, ...)`

返回一个 bytes 对象，其中包含根据格式字符串 format 打包的值 v1, v2, ... 参数个数必须与格式字符串所要求的值完全匹配。

`struct.unpack(format, buffer)`

根据格式字符串 format 从缓冲区 buffer 解包（假定是由 pack(format, ...) 打包）。 结果为一个元组，即使其只包含一个条目。 缓冲区的字节大小必须匹配格式所要求的大小，如 calcsize() 所示。