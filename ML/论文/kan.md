
## demo

https://github.com/KindXiaoming/pykan/blob/master/tutorials/Example_1_function_fitting.ipynb

## vs lgbm in simple scene

```python
from kan import KAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)

X_train, X_test, y_train, y_test = train_input, test_input, train_label, test_label
# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, y_train)
validation_data = lgb.Dataset(X_test, y_test)

dataset = {}
dataset['train_input'] = torch.from_numpy(train_input)
dataset['test_input'] = torch.from_numpy(test_input)
dataset['train_label'] = torch.from_numpy(train_label[:,None])
dataset['test_label'] = torch.from_numpy(test_label[:,None])

# X = dataset['train_input']
# y = dataset['train_label']
# plt.scatter(X[:,0], X[:,1], c=y[:,0])
```

kan :

```python
model = KAN(width=[2,1], grid=3, k=3)

def train_acc():
    return torch.mean((torch.round(model(dataset['train_input'])[:,0]) == dataset['train_label'][:,0]).float())

def test_acc():
    return torch.mean((torch.round(model(dataset['test_input'])[:,0]) == dataset['test_label'][:,0]).float())

results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc));

results['train_acc'][-1], results['test_acc'][-1]
```

lightgbm :

```python
# 参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'multiclass',  # 目标函数 poisson/quantile/quantile_l2/gamma/binary/multiclass/regression
    'num_class': 2,           # 类别数

    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息

    'nthread': -1, # 线程数量， -1 表示全部线程 
}

# 模型训练
gbm = lgb.train(params, train_data, num_boost_round=2000, valid_sets=[validation_data], callbacks=[lgb.early_stopping(5)])

# 模型预测
y_pred_probs = gbm.predict(X_test)
y_pred = [np.argmax(line) for line in y_pred_probs]

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
print('The accuracy of prediction is:', accuracy)
```







L-BFGS（Limited-memory Broyden-Fletcher-Goldfarb-Shanno）是一种用于无约束优化问题的迭代算法，特别适用于大规模问题。它是BFGS算法的变种，BFGS是一种准牛顿法，用于近似目标函数的二阶导数（即Hessian矩阵），从而加速收敛。

L-BFGS的主要特点在于其“有限记忆”特性。传统的BFGS算法需要存储和操作完整的Hessian矩阵，这在大规模问题中会导致内存和计算资源的过度消耗。L-BFGS通过仅存储和使用最近几次迭代的信息（通常是梯度和位置的变化），显著减少了内存需求和计算复杂度。

具体来说，L-BFGS算法的步骤如下：

1. **初始化**：选择初始点 \\( x_0 \\) 和初始近似Hessian矩阵（通常为单位矩阵）。
2. **迭代**：
   - 计算当前点的梯度 \\( g_k \\)。
   - 使用有限记忆的信息（如最近的梯度和位置变化）更新近似Hessian矩阵。
   - 计算搜索方向 \\( p_k \\)，通常通过近似Hessian矩阵和当前梯度的乘积得到。
   - 进行线搜索，找到合适的步长 \\( \\alpha_k \\)，使得目标函数在新点 \\( x_{k+1} = x_k + \\alpha_k p_k \\) 处有足够的下降。
   - 更新位置 \\( x_{k+1} \\) 和梯度 \\( g_{k+1} \\)。
3. **收敛判断**：如果梯度的范数或目标函数的变化量小于预设的阈值，则停止迭代。

L-BFGS在许多实际应用中表现出色，特别是在机器学习和数据科学领域的大规模优化问题中，如训练大型神经网络和支持向量机。其高效性和较低的内存需求使其成为处理大数据集和高维问题的理想选择。

-------------

符号回归（Symbolic Regression）是一种数据建模技术，用于揭示数据间的数学关系。它与传统回归分析不同，因为它不预设模型，而是从一组算术运算、变量、常数中自动生成数学表达式（模型）。

主要步骤包括：
1. **产生表达式**：利用遗传编程（genetic programming）随机生成初始方程。
2. **评估**：计算方程与实际数据点的误差。
3. **选择和改进**：通过遗传操作（如交叉、变异）来选择并产生新一代表达式。

优点是灵活性高，可以找到非常复杂的关系；缺点是计算量大，可能需要长时间来找寻最佳模型。应用场景包括量化金融、系统控制、生物信息学等领域。


-------------

看着更多是 符号数学，，


-----------

参考资料：
- chatgpt