

`Pandas` 是 Python 的**核心数据分析支持库**，是 Python 中统计计算生态系统的重要组成部分。

```python
import pandas as pd
import numpy as np

# 更改显示情况
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

pd.set_option('display.float_format',lambda x : '%.4f' % x)
```



</br>

## _数据结构_

1. **Series**

Series 操作类似字典类型，保留 `in` 操作、`.get(key,default=None)` 方法

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8])
```


2. **DataFrame**


```python
df = pd.DataFrame()

# Empty DataFrame
# Columns: []
# Index: []

# 从 list 中创建
df = pd.DataFrame([[k,v] for k,v in d.items()], columns=['k','v'])

# 从 dict 中创建
df = pd.DataFrame({
    'col1': [],
    'col2': []
})
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

## _查看&行遍历&排序_

重要方法： head(), info(), describe()


```python
# 查看首尾几行数据
df.head(10)
df.tail()

# 查看每列的数据类型
df.dtypes

# 随机抽取 100 元素
df.sample(100)
# 抽取 20%
df.sample(frac=0.2)
# 不放回抽样
df.sample(n=2, replace=True)

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

# 查看特定的列
cols = ['a', 'b']
df[cols]
```

```python
for index, row in pddf.iterrows():
    a,b,c = row.to_list()
```


```python
# 根据索引排序，默认升序
df.sort_index(axis=0,ascending=True)

# 根据cnt值降序排列
df.sort_values(by='score_cnt', ascending=False)

df.sort_values(axis=0,ascending=True)

# 按时间升序，以显示
res = pddf.sort_values(by='CREATED_AT')
res
```


使用各种查看方法，能更快新建 dataframe：

```python
# 新建一个 dataframe 为前100行
new_df = df.head(100)

df2 = df.copy()

# 新增一列
df['idx'] = range(len(df))
```


</br>

## _选择&常见操作_

（1）获取单列，`df.a` 与 `df['a']` 等效

```python
df['w']                 # get w col
df['w'].unique()        # # of distinct values in a column
```

（2）获取行，用 [ ] 切片行，如 `df[1:]`


（3）`df.iloc()` 通过整数位置来访问和选择数据

```python
# 选取 10-20 行
df.iloc[10:20]

# 选取第 1，2，5 列
df.iloc[:, [1,2,5]]

df.loc[:, ["A", "B"]]
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

# 选取 daraframe 中 ratio 最大、最小的行
max_row = fres.loc[fres['ratio'].idxmax()]
min_row = fres.loc[fres['ratio'].idxmin()]

df2[df2["E"].isin(["two", "four"])]
```

转化为 list 进行操作

```python
x = raw2['_KEY'].to_list()
```

从data 中筛选不在 test 的数据：

```python
train = data[~data.index.isin(test.index.tolist())]
```


逐行替换部分 columns, 构造新 dataframe

```python
new_df = pd.DataFrame(columns=feas) 


for i, row in online_user_df.iterrows():
    if i%500 == 0:
        print(i)
    score = int(row[0])
    d = eval(row[1].replace('null', 'None'))
    
    selected_row ={}
    tmp = offline_df[(offline_df['USER_ID'] == d['USER_ID'])&(offline_df['NEXT_SEED'] == d['SEED_ID'])&(offline_df['NEXT_SCORE'] == score)]
    if tmp.shape[0] == 1:
        for col in feas:
            if col in need_change_cols:
                selected_row[col] = tmp[col]
            else:
                selected_row[col] = d[col]


        new_df = new_df.append(selected_row, ignore_index=True)
```




</br>

_常见操作_

```python
# 查看分位数 
df['col'].quanile(0.5)

# 查看某一列 > 10 的数据
df[pd.to_numeric(df['col1']) > 10]

df2 = df[(20 < df['CLEAR_RATIO']) & (df['CLEAR_RATIO'] <= 75)]

# 再复杂一点
df['col2'] = abs(df['col1'] - 0.5)*df['col3']

# 分组再统计
seed_res = data_oldusers_test.groupby('NEXT_SEED')['bias'].agg([
    ('win_ratio', lambda x: (x > 0).mean()), ('score_cnt', 'size')
])

# 类型转化
data['NEXT_SCORE'] = data['NEXT_SCORE'].astype('int')
```


[groupby 官方文档](https://pandas.pydata.org/docs/user_guide/groupby.html)

[Windowing operations 官方文档](https://pandas.pydata.org/docs/user_guide/window.html)

```python
for window in s.rolling(window=2):
    print(window)
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

## _自定义函数_

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



# eg3 耗时较久
from bisect import bisect_right

def func(x):
    seed, score = x['SEED'], x['SCORE']
    if seed[-2:] in ['Y1', 'Y2', 'Y3']:
        return bisect_right(score, [0, 5000, 11000, 21000, 36000])
    else:
        return bisect_right(score, [0, 5000, 18000, 30000, 56000])
    
    

df['LABEL'] = df.apply(func, axis=1) # 按行应用


eg. 根据 df 生成提示词：
template = "\n\nCategory:\nkaggle-{Category}\n\nQuestion:\n{Question}\n\nAnswer:\n{Answer}"

df["prompt"] = df.progress_apply(lambda row: template.format(Category=row.Category,
                                                             Question=row.Question,
                                                             Answer=row.Answer), axis=1)
data = df.prompt.tolist()
```

</br>

## _空值&删除_


```python
# 删除存在 NA/null 数据的行
df.dropna()

# 利用 subset 来指定要检查的列
# inplace 表示在原始对象上直接修改
df.dropna(subset= ['NAME'], inplace = True)

#检查空值
pd.isnull(object)

# 填充所有列/单列
df = df.fillna(0)
df['col1'] = df['col1'].fillna(0)


df = df.replace(r'\\N',np.nan)

# 计算非零比例
np.count_nonzero((y_pred>=y_test)) / y_test.shape[0]


# 删除特征
# axis 0 表示行，1 表示列
df.drop('feature_variable_name', axis=1)
```


移位：

```python
df.shift(2)
# 除索引外的列，往下移两位，默认用 NaN 填充缺失的
```




</br>

## _连接&合并_

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


</br>

## _文件读写_


```python
# pip install openpyxl

df = pd.read_csv('filename.csv')
df = pd.read_excel('filename.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

df.to_csv('filename.csv', index=True)
df.to_excel('filename.xlsx', sheet_name='Sheet1')
```

分块读取 csv

```python
import pandas as pd

chunksize = 10**4  # 分块大小，此处例子按每10000行划分
filename = 'large_file.csv'

for chunk in pd.read_csv(filename, chunksize=chunksize):
    # 处理每块数据
    pass
```





--------

使用原生方法读取 csv 文件：

```python
i = 1
with open('citus.csv') as f:
    for line in f.readlines()[1:]:
        print(line.split(","))
        i += 1
        if i== 10:
            break
```

但读取出来，换行等符号都在，很多不方便；使用 csv 模块：


```python
import csv

i = 0
with open('citus.csv') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        print(row)
        i += 1
        if i == 10:
            break
```



demo, 使用 csv 充当中转进行数据迁移

使用 csv 要比 df.iterrows() + row.to_list() 快很多（在如下csv导到postgre过程）

关键函数：
- with open(csv_file) as f
- reader = csv.reader(f)
- header = next(reader)
- for row in reader



```python
import psycopg
from rich.progress import Progress
import pandas as pd
from rich.progress import track

# session 从远端1 获取数据并分成10份保存
df = session.sql(SQL).to_pandas()

num_rows = df.shape[0]

chunk_size = -(-num_rows//10)

for i in range(10):
    start_row = i*chunk_size
    end_row = start_row + chunk_size
    df_part = df.iloc[start_row:min(end_row, num_rows)]
    csv_file = f'part{i+1}.csv'
    df_part.to_csv(csv_file, index = False) 


# 连接 远端2 将数据写入
with psycopg.connect(PG_CONFIG) as conn:
    with conn.cursor() as cur:

        for i in range(10):

            csv_file = f'part{i+1}.csv'
            with open(csv_file) as f:
                reader = csv.reader(f)
                header = next(reader)
                # df = pd.read_csv(csv_file)
                # task = progress.add_task(f"{csv_file}", total=df.shape[0])
                with cur.copy("COPY scheme_name.table_name (user_id, created_at) FROM STDIN") as copy:
                    for row in track(reader,description=f"{csv_file}"):
                        copy.write_row(row)
                        # progress.update(task, advance=1)
```





</br>

## _demo_

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

eg3. 从 psycopg 提取数据构造 dataframe

```python
def main(schema, table): 
    query_match = f"""
    select *
    from {schema}.{table}
    limit 100
    """

    query_format = f"""
    SELECT
        column_name,
        data_type
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE "table_schema" = '{schema}' and "table_name" = '{table}';
    """
    with (
        psycopg.connect(PG_CONFIG) as conn,
        conn.cursor() as cur
    ):
        cur.execute(query_format)
        res = cur.fetchall()
        COL = [i[0] for i in res]
        
        cur.execute(query_match)
        res = cur.fetchall()
        df = pd.DataFrame([i for i in res], columns=COL)
    return df

main('s1', 't1')
```

可以使用 cursor.description 来简化上述步骤


```python
with (
    psycopg.connect(PG_CONFIG) as conn,
    conn.cursor() as cur
):
    cur.execute(query_match)
    res = cur.fetchall()
    column_name_list = [desc[0].upper() for desc in cursor.description]
    df = pd.DataFrame([i for i in res], columns=column_name_list)
```







-----------

参考资料：
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- Pandas中文网：https://www.pypandas.cn/
- Pandas练习：https://zhuanlan.zhihu.com/p/69371799
- [23种pandas操作](https://zhuanlan.zhihu.com/p/43018099)
- chatgpt
