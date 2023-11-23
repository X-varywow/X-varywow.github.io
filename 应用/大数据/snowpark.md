
## _preface_

snowpark提供了一个直观的 API，用于查询和处理数据管道中的数据。
使用此库，您可以构建在 Snowflake 中处理数据的应用程序，而无需将数据移动到运行应用程序代码的系统。

相比 snowflake connector，snowpark优势：
- supports pushdown for all operations, including Snowflake UDFs
- not require a separate cluster outside of Snowflake for computations. All of the computations are done within Snowflake.


</br>

## _构建流程_

本地构建：

```bash
# 使用官方文档中的 3.8 会有个 cffi 报错
conda create --name py310_env --override-channels -c https://repo.anaconda.com/pkgs/snowflake python=3.10 numpy pandas

conda activate py310_env

pip install snowflake-snowpark-python

pip install "snowflake-snowpark-python[pandas]"

pip install notebook

jupyter notebook
```

sagemaker 中构建：

```bash
source activate

conda create --name py310_env --override-channels -c https://repo.anaconda.com/pkgs/snowflake python=3.10 numpy pandas

conda activate py310_env

pip install snowflake-snowpark-python

pip install "snowflake-snowpark-python[pandas]"


# 有个 LibraryNotFoundError: Error detecting the version of libcrypto 报错，，

# 参考：https://github.com/wbond/oscrypto/issues/75
pip uninstall oscrypto -y

pip install git+https://github.com/wbond/oscrypto.git@d5f3437ed24257895ae1edd9e503cfb352e635a8

pip install ipykernel

# 完成
python -m ipykernel install --user --name=snow_park_env
```





</br>

## _使用方法_

（1）建立 session 连接：

```python
from snowflake.snowpark import Session

snowflake_config = {
    'user': ...
    'password': ...
    ...
}

session = Session.builder.configs(snowflake_config).create()
```

（2）获取数据并进行操作

```python
from snowflake.snowpark.functions import col


# Create a DataFrame that contains the id, name, and serial_number
# columns in the “sample_product_data” table.
df = session.table("sample_product_data").select(
    col("id"), col("name"), col("name"), col("serial_number")
)

# Show the results 
df.show()


# 方式二：
df = session.sql("select * from tb1 limit 10")

# 可直接转为本地 pandas.dataframe（注意大小）
new_df = df.to_pandas()
```


（3）数据写入 snowflake

```python
data = pd.DataFrame([[var_name, a.mu, a.sigma, a.ordinal()] for var_name, a in d.items()])
data.rename(columns={0:'var_name', 1:'mu', 2:'sigma', 3:'rank'},inplace=True)

# 对于 pandas.dataframe 格式数据：
session.createDataFrame(data).write.save_as_table(table_name='snowflake_table_name', mode='overwrite')

# 对于 snowpark.dataframe 格式数据：
snowpark_df.write.save_as_table(table_name='snowflake_table_name', mode='overwrite')
```


```python
session.close()
```


</br>

## _语法说明_

snowpark.table.Table

```python
from snowflake.snowpark.functions import col

df = session.table("   ")
df = df.filter(col('user_id') == 14329517)

df.show()
```



snowpark.dataframe.DataFrame

```python
df = session.sql()

```



snowflake.snowpark.row.Row

```python
df = session.sql("    ").collect()

type(df)         # -> list
type(df[0])      # -> snowflake.snowpark.row.Row

df[0].as_dict().keys()  # col keys
```



snowflake.snowpark.functions



```python
df = session.table("  ")

df.count()

df = df.filter(F.col("entry_kind") == 2)

df.count()

# filter 等操作不会同步到 snowflake 远端
```

更多的数据转换方法：

https://medium.com/snowflake/your-cheatsheet-to-snowflake-snowpark-dataframes-using-python-e5ec8709d5d7



</br>

## _实例：向 citus 集群导数据_

```python
import psycopg
from rich.progress import Progress
import pandas as pd

PG_CONFIG = "host=,, port=,, dbname=,, passward=,"

SQL = f"""
select * from t1
"""
statement = session.sql(SQL)

n = 0
for i, df in enumerate(statement.to_pandas_batches()):
    batch_file_name = f"batch_{i}.csv"
    df.to_csv(batch_file_name, index=False)
    n = i

with (
    psycopg.connect(PG_CONFIG) as conn,
    conn.cursor() as cur,
    cur.copy("COPY target_table (col1, col2) FROM STDIN") as copy,
    Progress() as progress
):
    task = progress.add_task("[red]Copy into Citus", total=n+1)
    for i in range(n):
        csv_file = f'batch_{i}.csv'
        with open(csv_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in track(reader, description=f"{csv_file}"):
                copy.write_row(row)
        progress.update(task, advance=1)
```





</br>

## _other_



```python
dataFrame = session.table(table_name).filter(col['col1'] == val)

session.add_packages("numpy", "pandas", "xgboost==1.5.0")

session.add_requirements("mydir/requirements.txt") 
```

```python
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)

df = session.table("tb_name")
df.show()
```


```python
# 对于运算量较小、且逻辑相对复杂的操作，snowpark 是具有优势的
df = session.sql(SQL)
pddf = df.to_pandas()
```

```python
# 不进行 collect() 会是一个 snowpark.dataframe 类型
df = session.sql("select * from t1 limit 10;").collect()

type(df)         # -> list
type(df[0])      # -> snowflake.snowpark.row.Row


df[0].as_dict().keys()  # col keys
```




---------

参考资料：
- https://docs.snowflake.com/en/developer
- https://docs.snowflake.com/en/developer-guide/snowpark/index.html
- [Getting Started with Data Engineering and ML using Snowpark for Python](https://quickstarts.snowflake.com/guide/getting_started_with_dataengineering_ml_using_snowpark_python/index.html) ⭐️
- [大佬 snowpark 使用过程](https://github.com/Snowflake-Labs/sfguide-getting-started-machine-learning/blob/main/hol/1_2_SOLUTION_basic_data_exploration_transformation.ipynb) ⭐️
- [Using Snowpark for Python with Amazon SageMaker](https://medium.com/snowflake/using-snowpark-for-python-with-amazon-sagemaker-44ec7fdb4381)