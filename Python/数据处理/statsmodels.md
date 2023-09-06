

python中专门用于统计学分析的包


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
 
distance = [0.7, 1.1, 1.8, 2.1, 2.3, 2.6, 3, 3.1, 3.4, 3.8, 4.3, 4.6, 4.8, 5.5, 6.1]
loss = [14.1, 17.3, 17.8, 24, 23.1, 19.6, 22.3, 27.5, 26.2, 26.1, 31.3, 31.3, 36.4, 36, 43.2]
data = pd.DataFrame({'distance':distance, 'loss':loss})


y1 = loss
X1 = distance
X1 = sm.add_constant(X1)      #增加一个常数1，对应回归线在y轴上的截距

regression1 = sm.OLS(y1, X1)  # 最小二乘法 Ordinary Least Squares
model1 = regression1.fit() 

model1.params
```

```python
model1.summary()
```




-----------

参考资料：
- [官方文档](https://www.statsmodels.org/stable/index.html)