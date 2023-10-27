

`Pandas` 是 Python 的**核心数据分析支持库**，是 Python 中统计计算生态系统的重要组成部分。

```python
import pandas as pd
import numpy as np

# 更改显示情况
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
```



</br>

## _数据结构_

- Series

```python
pd.Series([1,2,3]，index=[0,1,2])

# 0    1
# 1    2
# 2    3
# dtype: int64
```

- DataFrame

```python
df = pd.DataFrame()

# Empty DataFrame
# Columns: []
# Index: []

pandas.DataFrame(
    data, 
    columns=['distance', 'collision_cos_theta', 'spin_x', 'spin_y', 'strength']
)
```

```python
dates=pd.date_range('20200511',periods=3)
dates

# DatetimeIndex(['2020-05-11', '2020-05-12', '2020-05-13'], dtype='datetime64[ns]', freq='D')
```

```python
pd.DataFrame(np.random.randn(3,4),index=dates,columns=list('abcd'))
```

|            | a         | b         | c         | d         |
| ---------- | --------- | --------- | --------- | --------- |
| 2020-05-11 | -1.712588 | 0.403376  | -0.152608 | -0.428465 |
| 2020-05-12 | -1.259988 | -0.310385 | -0.816578 | 0.321397  |
| 2020-05-13 | -0.444678 | -1.894342 | 0.172485  | 0.717187  |





</br>

## _查看&行列_

重要方法： head(), info(), describe()


```python
# 查看首尾几行数据
df.head(10)
df.tail()

# 查看行号
df.index
# 查看列名
df.columns

# 查看某一列唯一实体
data["col_name"].unique()


# 查看统计信息
df.describe()

df.shape   # (rows, cols)
df.info()  # more info than shape
```

```python
# of distinct values in a column
df['w'].unique()
```


利用各种查看方法，能更快新建 dataframe：

```python
# 新建一个 dataframe 为前100行
new_df = df.head(100)
```

设置行列名称：

```python
# 设置索引为 1~10
df.set_index(pd.RangeIndex(1, 11), inplace = True)

# 更改列名
df.columns=['grammer', 'score', 'cycle']

# 更改列名
df.rename(columns={0:'var_name', 1:'mu', 2:'sigma', 3:'rank'},inplace=True)
df.rename(columns = {df.columns[2]:'size'}, inplace=True)
```


reindex() 用于重排 Series 或 DataFrame 对象

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Tokyo']}

df = pd.DataFrame(data)

new_index = ['A', 'B', 'C']
new_columns = ['Name', 'Age', 'City', 'Country']

new_df = df.reindex(index=new_index, columns=new_columns, fill_value='Unknown')

print(new_df)
```



</br>

## _选择&常见操作_

（1）获取单列，`df.a` 与 `df['a']` 等效


（2）获取行，用 [ ] 切片行，如 `df[1:]`



（3）`df.iloc()` 通过整数位置来访问和选择数据

```python
# 选取 10-20 行
df.iloc[10:20]

# 选取第 1，2，5 列
df.iloc[:, [1,2,5]]
```

（4）`df.loc()` 通过标签来访问和选择数据

```python
# 选取 index = 1 这一行数据
df.loc[1]

# 选择 x2-x4 列
df.loc[:, 'x2':'x4']

# 按条件选取
df.loc[df['a']>10, ['a','c']]

#将所有列倒序排列
df_desc = df.iloc[:, ::-1]
```

其它方法：
```python
# 最大、最小
df.nlargest(n, 'value')
df.nsmallest(n, 'value')
```

转化为 list 进行操作

```python
x = raw2['_KEY'].to_list()
```


- `Series`操作类似字典类型，含：保留字`in`操作、`.get(key,default=None)`方法


### 自定义函数

```python
# eg1
df["height"] = df["height"].apply(lambda x: 2 * x)

# eg2
def age_fun(x):
    if x < 12: return "children"
    elif x < 18: return "teenager"
    elif x < 65: return "audlt"
    else: return "old"

train["Age"] = train["Age"].apply(age_fun)
```

### 空值&删除


```python
# 删除存在 NA/null 数据的行
df.dropna()

# 利用 subset 来指定要检查的列
# inplace 表示在原始对象上直接修改
df.dropna(subset= ['NAME'], inplace = True)

#检查空值
pd.isnull(object)

# 填充
df.fillna(value)

df = df.replace(r'\\N',np.nan)

# 计算非零比例
np.count_nonzero((y_pred>=y_test)) / y_test.shape[0]


# 删除特征
# axis 0 表示行，1 表示列
df.drop('feature_variable_name', axis=1)
```



### 运算&排序 

1. **算术运算**根据行列索引，补齐后运算，运算默认产生浮点数。
2. 补齐时缺项填充NaN(空值)
3. 不同维度数据间运算为广播运算
4. `+` 或 `b.add(a,fill_value=NaN)`
5. **比较运算**


```python
# 根据索引排序，默认升序
df.sort_index(axis=0,ascending=True)

# 根据值排序
df.sort_values(axis=0,ascending=True)

# 按时间升序，以显示
res = pddf.sort_values(by='CREATED_AT')
res
```

groupby() shift()









### 连接&合并

```python
import pandas as pd

# 创建示例数据
df1 = pd.DataFrame({'key1': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key2': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# 根据 key 进行关联
merged_df = pd.merge(df1, df2, left_on='key1', right_on='key2', how='left')

pd.merge(multi_data, multi_out, how='left', on = ['DT', 'USER_ID'])

# 在 df1 上新增 df2 的数据
merged_df['value2'].fillna(0, inplace=True)  # 将缺失值填充为 0
df1['value2'] = merged_df['value2']  # 新增 value2 列

print(df1)
```

### 移位

```python
df.shift(2)
# 除索引外的列，往下移两位，默认用 NaN 填充缺失的
```




### 文件读写

CSV ：
```python
# 读
data = pd.read_csv('filename.csv')

# 写
df.to_csv('filename.csv', index=True)
```


Excel ：

```python
df.to_excel('filename.xlsx', sheet_name='Sheet1')
pd.read_excel('filename.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```



</br>

## _示例_

eg1. 新建几列，rank 值，与前一名分差，与第一名分差

```python
pddf['rank'] = None
pddf['d_next'] = None
pddf['d_max'] = None

for index, row in pddf.iterrows():
    arr = eval(row["RES"])[::-1]
    pddf.at[index, "RES"] = arr
    rank = arr.index(row["SCORE"])
    next_rank = max(0, rank-1)
    max_rank = 0
    pddf.at[index, "rank"] = rank + 1
    pddf.at[index, "d_next"] = arr[next_rank] - arr[rank]
    pddf.at[index, "d_max"] = arr[max_rank] - arr[rank]

pddf
```

eg2. 机器学习中数据集划分

```python
from sklearn.model_selection import train_test_split
import pandas as pd

X = df.drop("SCORE", axis = 1)
y = df["SCORE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)

# 对 label 编码
from sklearn.preprocessing import LabelEncoder
user_id_encoder = LabelEncoder()
user_id_encoder.fit_transform(train_df['USER_ID'])
train_df['USER_ID'] = seed_id_encoder.transform(train_df['USER_ID'])




train_df.to_csv('train.csv', index=True)
```



-----------

参考资料：
- Pandas中文网：https://www.pypandas.cn/
- Pandas练习：https://zhuanlan.zhihu.com/p/69371799
- [十分钟入门 Pandas](https://www.pypandas.cn/docs/getting_started/10min.html)
- [23种pandas操作](https://zhuanlan.zhihu.com/p/43018099)
- chatgpt
