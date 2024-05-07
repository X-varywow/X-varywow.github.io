
## _常见指标_

TP -> 正确的正例

FN -> 错误的负例

FP -> 错误的正例

TN -> 正确的负例


-------------------

准确率，Precision，判为对的中有多少个正例。
$$P=\frac{TP}{TP+FP}$$

召回率，Recall，正例有多少个被判为对了
$$R=\frac{TP}{TP+FN}$$

F1值
$$F1=\frac{2PR}{P+R} = \frac{2TP}{2TP+FP+FN}$$

-------------------

正确率 Accurary 
$$Accuracy=\frac{tp+tn}{all}$$

错误率 Error 
$$Error=\frac{fn+fp}{all}$$


-------------------

MSE，均方误差（Mean Squared Error）
$$ MSE =  \frac 1N \sum_{i=1}^N(y_i^2-\hat{y}_i^2)$$

RMSE，均方根误差（Root Mean Squared Error）
$$ RMSE = \sqrt {\frac 1N \sum_{i=1}^N(y_i^2-\hat{y}_i^2)}$$

--------------

MAE，平均绝对误差（Median Absolute Error）

$$ MAE = \frac 1n \sum_{i=1}^n|\hat{y_i}-y_i|$$

MAPE，平均绝对百分比误差（Mean Absolute Percentage Error）

$$ MAPE = \frac 1n \sum_{i=1}^n|\frac{\hat{y_i}-y_i}{y_i}|$$

-------------

梯度下降时，MSE 较 MAE 更为准确，异常值处 MAE 较 MSE 更加鲁棒

Huber Loss， 一定程度上结合了 MSE 和 MAE 的优点


$$L_{delta}(y, f(x)) = \begin{cases}
\frac12(y-f(x))^2, \quad \quad \quad if |y-f(x)| \le \delta \\
\delta|y-f(x)| - \frac12\delta^2, \quad if |y-f(x)| > \delta
\end{cases}$$

-------------------

分位数损失函数 Quantile loss function


$$L_{delta}(y, f(x)) = \begin{cases}
\alpha(y-f(x)) \quad \quad \quad  \quad \quad if \quad y>f(x) \\
(1-\alpha)(f(x)-y) \quad  \quad if \quad y<=f(x)
\end{cases}$$

($\alpha$ 表示分位数)

-------------------

交叉熵误差 cross entropy error

$$ E = -\sum_k t_k log y_k $$

（ $y_k$ 是神经网络的输出， $t_k$ 是正确解标签）

（交叉熵，用来衡量估计模型与真实概率分布之间的差异情况。）



-------------------


R2， 拟合系数（Coefficient of Determination），用来判断回归模型的解释力。

$$ R^2 = 1-\frac{\sum_i(\hat{y_i}-y_i)^2}{\sum_i(y_i-\overline {y})^2}$$

其中 残差平方和：$SS_{res} = \sum_i(\hat{y_i}-y_i)^2$

</br>

## _其他指标_

Pearson 相关系数

Kappa 系数

Spearman 相关系数， 斯皮尔曼相关系数



</br>

## _常见距离_


欧氏距离，曼哈顿距离，切比雪夫距离，闵可夫斯基距离，马氏距离，余弦相似度，皮尔逊相关系数，

汉明距离，杰卡德相似系数，编辑距离，KL 散度，还有改进KL散度的对称的JS 散度

----------

Wasserstein距离，两个概率分布之间差异的度量

假定要把概率分布 $p(x)$ 转变为 $q(x)$，设距离函数（转移成本）为 $d(x,y)$, 则：

$$\omega[p,q] = inf_{\gamma \in \prod [p, q]} \iint \gamma (x,y) d(x,y) dxdy$$






</br>

## _一些例子_

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error, r2_score
)

def regression_metrics(ground_truth, prediction):
    msg = f"""
    [MSE]: {mean_squared_error(ground_truth, prediction)}
    [RMSE]: {np.sqrt(mean_squared_error(ground_truth, prediction))}
    [MAE]: {mean_absolute_error(ground_truth, prediction)}
    [R2]: {r2_score(y_true=ground_truth, y_pred=prediction)}
    [MedianAE]: {median_absolute_error(ground_truth, prediction)}
    [MAPE]: {mean_absolute_percentage_error(ground_truth, prediction)}
    """
    return msg
```

```python
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred)
print(report)
```



--------------

参考资料：
- [说一说机器学习中TP、TN 、FP 、FN](https://blog.csdn.net/qq_28834001/article/details/102922993)
- https://zhuanlan.zhihu.com/p/353418080
- chatgpt
