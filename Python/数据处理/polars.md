


## preface

速度会快很多，自用的时候可以替代 pandas.

https://github.com/pola-rs/polars


--------------

`性能测试`

```python
%%timeit

import polars as pl

seed_cols = ['col1', 'col2']

seed_df = pl.read_csv('aaa.csv', has_header=False, new_columns=seed_cols)
```

pandas 28 ms ± 154 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

polars 2.43 ms ± 21.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

**耗时仅为 8.68%**



</br>

## demo1

`官方 demo`

```python
>>> import polars as pl
>>> df = pl.DataFrame(
...     {
...         "A": [1, 2, 3, 4, 5],
...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
...         "B": [5, 4, 3, 2, 1],
...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
...     }
... )

# embarrassingly parallel execution & very expressive query language
>>> df.sort("fruits").select(
...     "fruits",
...     "cars",
...     pl.lit("fruits").alias("literal_string_fruits"),
...     pl.col("B").filter(pl.col("cars") == "beetle").sum(),
...     pl.col("A").filter(pl.col("B") > 2).sum().over("cars").alias("sum_A_by_cars"),
...     pl.col("A").sum().over("fruits").alias("sum_A_by_fruits"),
...     pl.col("A").reverse().over("fruits").alias("rev_A_by_fruits"),
...     pl.col("A").sort_by("B").over("fruits").alias("sort_A_by_B_by_fruits"),
... )
shape: (5, 8)
┌──────────┬──────────┬──────────────┬─────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ fruits   ┆ cars     ┆ literal_stri ┆ B   ┆ sum_A_by_ca ┆ sum_A_by_fr ┆ rev_A_by_fr ┆ sort_A_by_B │
│ ---      ┆ ---      ┆ ng_fruits    ┆ --- ┆ rs          ┆ uits        ┆ uits        ┆ _by_fruits  │
│ str      ┆ str      ┆ ---          ┆ i64 ┆ ---         ┆ ---         ┆ ---         ┆ ---         │
│          ┆          ┆ str          ┆     ┆ i64         ┆ i64         ┆ i64         ┆ i64         │
╞══════════╪══════════╪══════════════╪═════╪═════════════╪═════════════╪═════════════╪═════════════╡
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 4           ┆ 4           │
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 3           ┆ 3           │
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 5           ┆ 5           │
│ "banana" ┆ "audi"   ┆ "fruits"     ┆ 11  ┆ 2           ┆ 8           ┆ 2           ┆ 2           │
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 1           ┆ 1           │
└──────────┴──────────┴──────────────┴─────┴─────────────┴─────────────┴─────────────┴─────────────┘
```


```python
>>> df = pl.scan_csv("docs/assets/data/iris.csv")
>>> ## OPTION 1
>>> # run SQL queries on frame-level
>>> df.sql("""
...	SELECT species,
...	  AVG(sepal_length) AS avg_sepal_length
...	FROM self
...	GROUP BY species
...	""").collect()
shape: (3, 2)
┌────────────┬──────────────────┐
│ species    ┆ avg_sepal_length │
│ ---        ┆ ---              │
│ str        ┆ f64              │
╞════════════╪══════════════════╡
│ Virginica  ┆ 6.588            │
│ Versicolor ┆ 5.936            │
│ Setosa     ┆ 5.006            │
└────────────┴──────────────────┘
>>> ## OPTION 2
>>> # use pl.sql() to operate on the global context
>>> df2 = pl.LazyFrame({
...    "species": ["Setosa", "Versicolor", "Virginica"],
...    "blooming_season": ["Spring", "Summer", "Fall"]
...})
>>> pl.sql("""
... SELECT df.species,
...     AVG(df.sepal_length) AS avg_sepal_length,
...     df2.blooming_season
... FROM df
... LEFT JOIN df2 ON df.species = df2.species
... GROUP BY df.species, df2.blooming_season
... """).collect()
```


</br>

## demo2

`自用 demo`


demo1. json.loads() 扩充 df

```python
df = df.with_columns(
    pl.col("features").str.json_decode().alias("parsed")
).unnest("parsed")
```


------------

demo2. 读取csv，连接df 等

```python
import polars as pl

null_vals = ["\\N", "NA", "null"]

# 1. 读取，特殊值处理
df = pl.read_csv(
    'tmp.csv',
    has_header=False,
    new_columns=user_cols,
    null_values=null_vals
)

# 2. 将每列的值全部改成3
df = df.with_columns([
    pl.lit(3).alias('col1')
])

# 3. 列名大写
df = df.rename({col: col.upper() for col in df.columns})

# 4. 字符串去多余引号
df = df.with_columns([
    pl.col('col2').str.strip_chars('"').alias("col2")
])

# 5. 连接
df = df1.join(
    df2,
    left_on = ['col1', 'col2'],
    right_on = ['col3', 'col4'],
    how="inner"
)

# 6. 列转换，重命名
df = df.with_columns([
    pl.col("CREATED_AT").str.strptime(pl.Datetime).alias("CREATED_AT")
])

df = df.with_columns([
    pl.col("col1").str.strip_chars('"').cast(pl.Int64)
])

df = df.rename({"col1": "col2"})


# 7. 时间 filter
split_date = pl.date(2025, 5, 10)

data_train = df.filter(pl.col("CREATED_AT") < split_date)


# 8. 转成 pands; lightgbm 不知道为啥
data_train = data_train.to_pandas()
```




-------------

参考资料：
- https://docs.pola.rs/
- gpt
