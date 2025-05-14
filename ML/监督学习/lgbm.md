
## _LGBM_


### _基本原理_

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


--------------


在LightGBM中，`eval_metric`参数用于评估模型的性能，而不是用来指定训练过程中使用的损失函数。

模型训练时使用的损失函数是通过 `objective` 参数来指定的。

因此，如果你设置了`objective`为`quantile`（分位数回归），那么无论你在`eval_metric`中指定了哪些 **评估指标**（比如`'mae'`, `'huber'`等），模型训练时使用的损失函数都是分位数损失。

--------------

`eval_metric` 作用：（感觉只有观测和手动调整超参的作用）

1. 性能评估
- **监控训练过程**：`eval_metric`允许你在模型训练过程中监控一个或多个评估指标。这意味着你可以实时地看到模型在训练集和验证集上的表现，帮助你了解模型是否在学习，以及学习的速度如何。
- **模型比较**：通过在不同模型或不同参数设置下观察这些评估指标，你可以比较哪个模型或哪组参数能够更好地解决你的问题。

2. 过拟合检测
- **早停（Early Stopping）**：当设置了`eval_metric`和验证集时，你可以利用早停机制来防止过拟合。如果在一定数量的迭代中，选定的评估指标没有改善，训练过程可以提前停止。这有助于避免浪费计算资源，并且防止模型在训练数据上过度拟合。

3. 模型优化方向
- **指导参数调优**：通过观察不同的`eval_metric`表现，你可以对模型的参数进行调整，以优化特定的性能指标。例如，如果你关注模型的准确率，你可能会优先选择使准确率最大化的参数配置。

4. 适应特定任务需求
- **灵活选择评估指标**：不同的机器学习任务可能需要不同的评估指标。例如，对于分类问题，你可能会关注准确率、召回率或AUC等；对于回归问题，你可能会关注均方误差（MSE）、平均绝对误差（MAE）或R²等。`eval_metric`允许你根据任务的特性和需求选择最合适的评估指标。

5. 自定义评估指标
- **灵活性和扩展性**：如果内置的评估指标不能满足你的需求，许多框架（包括LightGBM）允许你定义自己的评估函数。这提供了极大的灵活性，使你能够针对特定的业务问题设计和优化模型。





### _原生方式_

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
    'objective': 'multiclass',  # 目标函数 poisson/quantile/quantile_l2/gamma/binary/multiclass/regression
    'num_class': 3,           # 类别数

    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息

    'nthread': -1, # 线程数量， -1 表示全部线程 

    # max_cat_threshold: 1024,
    # max_bin: 256,

    # 'lambda_l1': 0.1,
    # 'lambda_l2': 0.2,
    # 'max_depth': 4,
    
    # 'n_estimators': 2000, # 迭代轮次，默认 100， 通常设置较大并配合早停机制
    # 官方建议在 train 中 指定 num_boost_round
    
    # 'metric': {'l2', 'auc'},  # 评估函数

    # class_weight: 'balanced',
    # sample_weight: data_train['weight'].values,
}

# 模型训练
gbm = lgb.train(params, train_data, num_boost_round=2000, valid_sets=[validation_data], callbacks=[lgb.early_stopping(5)])

# 模型预测
y_pred_probs = gbm.predict(X_test)
y_pred = [np.argmax(line) for line in y_pred_probs]

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
print('The accuracy of prediction is:', accuracy)

# 保存/加载
gbm.booster_.save_model(model_name)
model = lgb.Booster(model_file = 'model-01')
```

多分类 objective：multiclass 需要额外设定  num_class = 5,

其它目标函数： regression_l1, regression_l2, quantile, poisson, mape

-------------


sklearn 接口形式，参考如下分数位回归：




### _分位数回归_

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




### _自定义目标函数_

```python
import numpy as np

def median_objective(y_true, y_pred) -> (np.ndarray, np.ndarray):
    # 计算残差
    residual = y_true - y_pred
    
    # 残差的一阶导数（梯度）
    # 对于中位数回归，梯度是残差的符号
    grad = np.where(residual > 0, -1, 1) if residual.size else np.zeros_like(residual)
    
    # 残差的二阶导数（Hessian）
    # 对于中位数回归，二阶导数是0，因为绝对值函数在任何点的曲率都是0
    hess = np.zeros_like(residual)
    
    return grad, hess
```



## 使用 optuna

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





## 使用 shap

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


其它方式查看特征重要性：

```python
# way 1
pd.DataFrame({
        'column': feas,
        'importance': model.feature_importances_,
}).sort_values(by = 'importance', ascending=False)

# way 2
lgb.plot_importance(model, max_num_features=20)

# way 3
# when 1 error: 'Booster' object has no attribute 'feature_importances_'
feature_importance = lgbm_model.feature_importance(importance_type='split')
feature_names = lgbm_model.feature_name()
importance_data = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

for feature, importance in importance_data:
    print(f"Feature: {feature}, Importance: {importance}")
```

-----------------

LightGBM的`plot_importance`函数通常使用基于模型内部的度量，例如增益（gain）或覆盖度（cover）来表示特征的重要性。增益衡量特征通过分裂增加的准确性，而覆盖度衡量该特征出现在多少样本的分布中。这些指标主要基于训练过程中树结构的信息。

相较之下，SHAP值（Shapley Additive Explanations）从博弈论的角度解释模型输出。通过考虑所有可能的特征组合，SHAP值为每个特征提供了其对模型最终预测的贡献程度，更全面地研究了特征值如何影响预测结果。

因此，虽然LightGBM的增益和SHAP值都用于评估特征重要性，但两者从不同的视角进行评估，可能会导致不同的重要性排名和解释。这种差异可能提供了对模型行为的更全面的理解。








## demo: lgbm&shap

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


--------

lgbm 14 分类模型实例：

```python
explain_df = pd.DataFrame([fixed_persona_default_values], columns=TOTAL_FEATURES)

explainer = shap.Explainer(model)

shap_values = explainer(explain_df)

# param2 means class; cur model : 0~ 13
shap_values = shap_values[..., 13]

shap.plots.bar(shap_values, max_display=20)

# help(shap.plots.bar)
```





## other

模型加载时间：14mb model 一次加载 90ms

降低模型大小：
set the histogram_pool_size parameter to the MB you want to use for LightGBM (histogram_pool_size + dataset size = approximately RAM used), lower num_leaves or lower max_bin

损失函数选择：MSE 比 MAE 更加重视较大的误差，如何不想让异常值过度地影响模型，MAE 会更好; 定好 objective 时已经指定了；

如果数据中存在较多的缺失且缺失的信息对目标变量有较重要影响，则应当使用错位填充的方法。

未经处理就直接喂给 LightGBM 让其自动应对缺失值，则会损失一些信息背景，从而无法最大程度提升模型的效果。


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


更好的模型效果：
- 常规的: 更大、更好的训练数据
- 较大的 max_bin
- 较小的 learning_rate

-----------

解释接口：

```python
!pip install graphviz
lgb.plot_tree(model)

lgb.plot_metric(model)

lgb.plot_importance(model)
```

查看树结构：

```python
num_trees = model.num_trees()
print(num_trees)


lgb.create_tree_digraph(model, tree_index=10)
```

-----------

!> 当测试集的数据分布发生大幅变化时，如值域改变，模型是不适用的。</br>
**尽量保证训练和测试的数据同分布**

树模型分裂点完全基于训练数据中观察到的范围，如果测试集中某个值超出范围，将无法做出有意义的判断；

值域改变问题中，刻度尺是基于训练数据的，学习到的特征重要性、树子节点、shap加加减减是拉不回结果的值域的。

许多机器学习模型（包括LightGBM）隐式或显式地假设训练数据和测试数据来自相同的或相似的概率分布 P(X, Y)


```python
import graphviz
graph = lgb.create_tree_digraph(model, tree_index=0)
graph.render(filename='tree', format='pdf')  # 生成 tree.pdf
```

查看第一棵树的结果，生成矢量图，可以看到决策树（yes-no 二叉树， 叶子结点为预测值）很呆




-----------


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
