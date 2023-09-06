
## _基本原理_

LightGBM（Light Gradient Boosting Machine）是一个 **实现GBDT算法的框架**。支持高效率的并行训练，并且具有 **更快的训练速度、更低的内存消耗**、更好的准确率、支持分布式可以快速处理海量数据等优点。


GBDT (Gradient Boosting Decision Tree) 是机器学习中一个长盛不衰的模型，其主要思想是 **利用弱分类器（决策树）迭代训练以得到最优模型**，该模型具有训练效果好、不易过拟合等优点。GBDT不仅在工业界应用广泛，通常被用于多分类、点击率预测、搜索排序等任务；在各种数据挖掘竞赛中也是致命武器，据统计Kaggle上的比赛有一半以上的冠军方案都是基于GBDT。



- 基于 Histogram 的决策树算法
- 带深度限制的 Leaf-wise 算法
- 单边梯度采样算法
- 互斥特征捆绑算法




## _常见用法_

```python
!pip install lightgbm
```

```python
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# 加载数据
iris = datasets.load_iris()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

# 参数
params = {
    'learning_rate': 0.1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': 4,
    'objective': 'multiclass',  # 目标函数
    'num_class': 3,
}

# 模型训练
gbm = lgb.train(params, train_data, valid_sets=[validation_data])

# 模型预测
y_pred = gbm.predict(X_test)
y_pred = [list(x).index(max(x)) for x in y_pred]
print(y_pred)

# 模型评估
print(accuracy_score(y_test, y_pred))
```

## _数据分析_

```python
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 查看 pandas.DataFrame
df.describe()

# 自动箱线图
df.boxplot()
plt.show()

# 相关性分析
df.corr()

# 相关性曲线
sns.pairplot(df, x_vars=[...], y_vars='strength', height=7, aspect=0.8, kind = 'reg')
plt.show()
```



## _使用 optuna_

超参数自动化调整

```python
import ...

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])
    if regressor_name == 'SVR':
        svr_c = trial.suggest_loguniform('svr_c', 1e-10, 1e10)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X, y = sklearn.datasets.load_boston(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
```

```python
import torch

import optuna

# 1. Define an objective function to be maximized.
def objective(trial):

    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, 10))
    layers.append(torch.nn.LogSoftmax(dim=1))
    model = torch.nn.Sequential(*layers).to(torch.device('cpu'))
    ...
    return accuracy

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```


## _使用 shap_

对机器学习模型进行解释，解释特征的贡献值等

```python
def lgbm_predict(df):
    train_x, test_x, train_y, test_y = train_test_split(df.iloc[:, :-1], df.strength, train_size = 0.8, test_size = 0.2)

    study = optuna.create_study(
        direction="minimize", 
        study_name=f"LGBM Regressor"
    )
    
    # objective function: return a value to be minimized
    #                     and learn the best params
    # so objective contain: params, train, eval, predict, and metric
    study.optimize(
        lambda trial: objective(trial, train_x, test_x, train_y, test_y), 
        n_trials=100
    )
    best_params = study.best_trial.params
    print(f"finish model optimization with best trial accuracy: {study.best_value}, best_params:{best_params}")

    model = lgb.LGBMRegressor(...)
    model.fit(...)
    predictions = model.predict(...)

    return train_x, model

X, model = lgbm_predict(df)

explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.plots.bar(shap_values)
```


--------------------

参考资料：
- [深入理解LightGBM](https://zhuanlan.zhihu.com/p/99069186)
- [万字详解：LightGBM 原理、代码最全解读！](https://zhuanlan.zhihu.com/p/447252042)
- [LightGBM官方文档](https://lightgbm.readthedocs.io/en/v3.3.2/)
- [Kaggle神器LightGBM最全解读！](https://cloud.tencent.com/developer/article/1758058)
- https://optuna.org/
- [optuna 文档](https://zh-cn.optuna.org/index.html)
- [用 SHAP 可视化解释机器学习模型实用指南(上)](https://mp.weixin.qq.com/s?__biz=Mzk0OTI1OTQ2MQ==&mid=2247500066&idx=1&sn=fe878ccbbd1299366ada3ec9f622a402&chksm=c3599c88f42e159eef4da04751df3ed93aa3a0d53ad4d07c1a06036a9cd0bbb85c011afaa82d&scene=21#wechat_redirect)
- [用 SHAP 可视化解释机器学习模型实用指南(下)](https://cloud.tencent.com/developer/article/1888981)