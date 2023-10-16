

`Pandas` 是 Python 的**核心数据分析支持库**，是 Python 中统计计算生态系统的重要组成部分。

```python
import pandas as pd
import numpy as np

# 更改显示情况
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 按时间升序，以显示
res = pddf.sort_values(by='CREATED_AT')
res


data = pd.read_csv("train.csv")

#see the first 10 rows
data.head(10)

#check the rows and cols
data.shape
data.info()

#print the name of all the columns
data.columns
#
len

#index
data.index

#检查空值
pd.isnull(object)

# 查看数据统计信息
data.describe()

# 查看某一列唯一实体
data["col_name"].unique()

# 删除特征
df.drop('feature_variable_name', axis=1)
#axis 0 表示行，1 表示列

# 通过特征名取数据
df.loc[feature_name]

# 对 DataFrame 使用函数
df["height"] = df["height"].apply(lambda x: 2 * x)

# 重命名行
df.rename(columns = {df.columns[2]:'size'}, inplace=True)
```

```python
# 设置索引为 1~10
df.set_index(pd.RangeIndex(1, 11), inplace = True)

# 更改列名
df.columns=['grammer', 'score', 'cycle']
```

```python
#将所有列倒序排列
df_desc = df.iloc[:, ::-1]
```


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









## 数据结构

- 一维的 `Series`
- 二维的 `DataFrame`

```python
pd.Series([1,2,3]，index=[0,1,2])

# 0    1
# 1    2
# 2    3
# dtype: int64
```

```python
pd.DataFrame()

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

## 输入输出

CSV ：
- `df.to_csv('filename.csv')`
- `pd.read_csv('filename.csv')`

Excel ：
- `df.to_excel('filename.xlsx', sheet_name='Sheet1')`
- `pd.read_excel('filename.xlsx', 'Sheet1', index_col=None, na_values=['NA'])`


## 查看
- `df.head()`，k值默认为5.
- `df.tail()`，k值默认为5
- `df.index`
- `df.columns`
- `df.describe()`，查看数据的分布情况
- `df.info()`

```python
# of distinct values in a column
df['w'].unique()
```


## 选择 & 常见操作
- 获取单列，`df.a`与`df['a']`等效
- 获取行，用 [ ] 切片行，如 `df[1:]`
- 按标签获取
- 按位置获取


```python
# 选取 10-20 行
df.iloc[10:20]

# 选取第 1，2，5 列
df.iloc[:, [1,2,5]]

# 选择 x2-x4 列
df.loc[:, 'x2':'x4']

# 按条件选取
df.loc[df['a']>10, ['a','c']]

# 最大、最小
df.nlargest(n, 'value')
df.nsmallest(n, 'value')
```

- `Series`操作类似字典类型，含：保留字`in`操作、`.get(key,default=None)`方法
- `.reindex()`，改变或重排`Series`或`DataFrame`索引数据输入输出
- `.drop()`，删除`Series`或`DataFrame`指定行或列索引，默认0轴(竖的)
- `apply()`

```python
def age_fun(x):
    if x < 12: return "children"
    elif x < 18: return "teenager"
    elif x < 65: return "audlt"
    else: return "old"

train["Age"] = train["Age"].apply(age_fun)
```


```python
# 删除存在 NA/null 数据的行
df.dropna()

# 填充
df.fillna(value)
```

## 运算 & 排序

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
```

-----------

参考资料：
- Pandas中文网：https://www.pypandas.cn/
- Pandas练习：https://zhuanlan.zhihu.com/p/69371799
- [十分钟入门 Pandas](https://www.pypandas.cn/docs/getting_started/10min.html)
- [23种pandas操作](https://zhuanlan.zhihu.com/p/43018099)
