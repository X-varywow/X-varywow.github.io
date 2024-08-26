
```python
from beautifultable import BeautifulTable

# 创建表格实例
table = BeautifulTable()

# 添加列标题
table.column_headers = ["Name", "Age", "Occupation"]

# 添加行数据
table.rows.append(["Alice", 28, "Engineer"])
table.rows.append(["Bob", 32, "Data Scientist"])

# 打印表格
print(table)
```