
## _基本原理_

LightGBM（Light Gradient Boosting Machine）是一个 **实现GBDT算法的框架**。支持高效率的并行训练，并且具有 **更快的训练速度、更低的内存消耗**、更好的准确率、支持分布式可以快速处理海量数据等优点。


GBDT (Gradient Boosting Decision Tree) 是机器学习中一个长盛不衰的模型，其主要思想是 **利用弱分类器（决策树）迭代训练以得到最优模型**，该模型具有训练效果好、不易过拟合等优点。GBDT不仅在工业界应用广泛，通常被用于多分类、点击率预测、搜索排序等任务；在各种数据挖掘竞赛中也是致命武器，据统计Kaggle上的比赛有一半以上的冠军方案都是基于GBDT。



- 基于 Histogram 的决策树算法
- 带深度限制的 Leaf-wise 算法
- 单边梯度采样算法
- 互斥特征捆绑算法

---------------

```python
!pip install lightgbm
```


- lightgbm.LGBMRegressor
- lightgbm.LGBMClassifier
- lightgbm.LGBMRanker








</br>

## _原生demo_

```python
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

# 加载数据
iris = datasets.load_iris()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, y_train)
validation_data = lgb.Dataset(X_test, y_test)

# 参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数 poisson/quantile/quantile_l2/gamma/binary/multiclass
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息

    # max_cat_threshold: 1024,
    # max_bin: 256,

    # 'lambda_l1': 0.1,
    # 'lambda_l2': 0.2,
    # 'max_depth': 4,
    # 'num_class': 3,

    # class_weight: 'balanced',
    # sample_weight: data_train['weight'].values,
}

# 模型训练
gbm = lgb.train(params, train_data, valid_sets=[validation_data], early_stopping_rounds=5)

# 模型预测
y_pred = gbm.predict(X_test)
y_pred = [list(x).index(max(x)) for x in y_pred]
print(y_pred)

# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# 保存/加载
gbm.booster_.save_model(model_name)
model = lgb.Booster(model_file = 'model-01')
```

sklearn 接口形式，参考如下分数位回归：



</br>

## _分位数回归_

quantile regression，最小化所选分位数切点产生的绝对误差之和

关心预测的期望值和潜在变异性，相当于间接把预测目标的分布给描述出来了，对于用户得分预测等场景很适用

```python
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

model = lgb.LGBMRegressor(
    task = 'train',
    objective = 'quantile',
    alpha = 0.5,  # 指定关心的分数位
    boosting_type = 'gbdt',
    learning_rate = 0.01,
    n_estimators = 2000,
    min_child_samples = 16,
    max_depth = 7,
    num_leaves = 127,
    random_state = 42,
    max_cat_threshold = 1024,
    max_bin = 256,
    class_weight = 'balanced'
)

callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]
model.fit(
    X_train, 
    y_train, 
    eval_set = [(X_train, y_train),(X_val, _val)],
    eval_metric = ['rmse', 'mape', 'huber'],
    callbacks = callbacks,
    feature_nmae = ,
    categorical_feature = 
)
```

```python
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# 加载数据
iris = load_iris()
data = iris.data
target = iris.target

# 划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 模型训练
gbm = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)

# 模型存储
joblib.dump(gbm, 'loan_model.pkl')
# 模型加载
gbm = joblib.load('loan_model.pkl')

# 模型预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# 特征重要度
pd.DataFrame({
        'column': feas,
        'importance': model.feature_importances_,
}).sort_values(by = 'importance', ascending=False)

lgb.plot_importance(model)


# 网格搜索，参数优化
estimator = LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
```


</br>

## _多分类_


```python
classifier = lgb.LGBMRegressor(
        task = 'train',
        objective = 'multiclass',
        num_class = 5,
        boosting_type = 'gbdt',
        # metric = 'multi_logloss',
        learning_rate = 0.01,
        n_estimators = 2000, 
        min_child_samples = 16, 
        max_depth = 7,  
        num_leaves = 32, 
        random_state = 42,
        max_cat_threshold = 1024,
        max_bin = 256,
        # class_weight = 'balanced',
         # sample_weight=data_oldusers_train_1['weight'].values,
    )

callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

classifier.fit(X_train, y_train, eval_set=[(X_train, y_train),
                                          (X_val, y_val)],
              # eval_metric=[
              #     'mae','huber'
              # ],
              callbacks=callbacks,
              feature_name=feas, 
              # no log
              # categorical_feature=categorical_features
             )
```





</br>

## _other_

更好的模型效果：
- 常规的: 更大、更好的训练数据
- 较大的 max_bin
- 较小的 learning_rate


----------------

其它目标函数： regression_l1, regression_l2, quantile, poisson, mape

特征工程、数据分析相关，参考：[Python/数据处理/特征工程](Python/数据处理/特征工程)


-----------

解释接口：

```python
!pip install graphviz
lgb.plot_tree(model)

lgb.plot_metric(model)

lgb.plot_importance(model)
```






</br>

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

- https://optuna.org/
- [optuna 文档](https://zh-cn.optuna.org/index.html)




</br>

## _使用 shap_

对机器学习黑盒模型进行解释，给出特征的贡献值等

更多参考：[ML/监督学习/工具包：shap](ML/监督学习/shap)



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


</br>


## _demo: lgbm&shap_

```python
import lightgbm as lgb
import shap
file_path = r'/home/ec2-user/SageMaker/'

# 1. load model
bst = lgb.Booster(model_file=f'{file_path}/model.pkl')
lgb.plot_importance(bst, max_num_features=20)

# 2. predict by feas
feas = ['user_id', 'score_100']
bst.predict(df.head(1)[feas])

# 中途有报错，指定一下 objective
bst.params['objective'] = 'quantile'

# 3. shap explain
explainer = shap.Explainer(bst)
x = df[df['difficulty_bin'.upper()] == 5].head(1)
shap_values = explainer(x)

# 4. waterfall plot
shap.plots.waterfall(shap_values[0])

# 5. batch plot
shap.plots.beeswarm(shap_values)
```

lgbm 只是对样本整体做了一个分位数回归，当样本整体需要看子样本维度的时候，如深水区、浅水区、都需要完美预测运动员的水平，但是分位数回归只是整体做了一个 50 分位的预测，

虽然整体上看偏差还是正态，无太大问题， 但是各个子样本（训练与预测）的分布不同、峰度、偏度不同，大概率会产生一定问题，可以尝试：拆分样本。


## other

模型加载时间：14mb model 一次加载 90ms

降低模型大小：
set the histogram_pool_size parameter to the MB you want to use for LightGBM (histogram_pool_size + dataset size = approximately RAM used), lower num_leaves or lower max_bin

损失函数选择：MSE 比 MAE 更加重视较大的误差，如何不想让异常值过度地影响模型，MAE 会更好。

查看模型特征：

```python
model.feature_name()
model.num_feature()
```


对模型精度影响的重要度：
1. 数据集
2. 损失函数
3. 超参数


--------------------

相关链接

[pandas](Python/数据处理/pandas)

[特征工程](Python/数据处理/特征工程)





--------------------

参考资料：
- [万字详解：LightGBM 原理、代码最全解读！](https://zhuanlan.zhihu.com/p/447252042)
- [LightGBM官方文档](https://lightgbm.readthedocs.io/en/v3.3.2/)
- https://caicaijason.github.io/2020/01/07/LightGBM%E7%AE%80%E4%BB%8B/ ⭐️
- https://www.showmeai.tech/article-detail/205 ⭐️ 具体应用向
- [LightGBM源码阅读+理论分析](https://mp.weixin.qq.com/s/XxFHmxV4_iDq8ksFuZM02w) 偏理论
- [为风控业务定制损失函数与评价函数（XGB/LGB）](https://cloud.tencent.com/developer/article/1557778)
