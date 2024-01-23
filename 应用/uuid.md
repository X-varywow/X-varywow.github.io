

UUID 是通用唯一标识符（Universally Unique Identifier）的缩写，用来让系统能够生成全球范围内唯一的标识符。

Python 中的 `uuid` 模块可以生成UUID。它提供了四种不同的生成UUID的方法：

1. `uuid.uuid1()`: 基于时间戳和MAC地址生成UUID，保证了在同一时空下的唯一性。
2. `uuid.uuid3(namespace, name)`: 基于名字的MD5散列值生成UUID。需要提供一个namespace和一个name。
3. `uuid.uuid4()`: 随机生成UUID，生成随机的128位数，有非常小的重复概率。
4. `uuid.uuid5(namespace, name)`: 基于名字的SHA-1散列值生成UUID。需要提供一个namespace和一个name。

这里有一个简单的例子，展示如何使用这个模块生成UUID:

```python
import uuid

# 基于时间和主机ID
uuid_one = uuid.uuid1()

# 基于名字和MD5哈希值
uuid_three = uuid.uuid3(uuid.NAMESPACE_DNS, 'example.com')

# 随机生成
uuid_four = uuid.uuid4()

# 基于名字和SHA-1散列值
uuid_five = uuid.uuid5(uuid.NAMESPACE_DNS, 'example.com')

print(f"UUID1: {uuid_one}")
print(f"UUID3: {uuid_three}")
print(f"UUID4: {uuid_four}")
print(f"UUID5: {uuid_five}")
```

`uuid`生成的UUID常用于数据库主键、分布式系统中标识唯一的事物或对象，以及任何需要不重复的ID的场景。


------------

参考资料：
- chatgpt