

SHAP(SHapley Additive exPlanations) 是一种解释机器学习模型决策的工具。

它基于博弈论中的 Shapley Value (一种公平分配合作游戏中合作产生的总收益的方法)。

对机器学习黑盒模型进行解释，给出特征的贡献度等

- 模型解释
- 特征重要性
- 依赖关系可视化

</br>

## _demo_

查看 lgbm 分类模型的特征重要度：

```python
import lightgbm as lgb
import shap
CLASSIFER = lgb.Booster(model_file = new_file_path)
CLASSIFER.params['objective'] = 'multiclass'

explainer = shap.TreeExplainer(CLASSIFER)
shap_values = explainer.shap_values(new_df.head(1)[new_feas])
shap.summary_plot(shap_values, new_df.head(1)[new_feas], plot_type="bar")
```
















示例 1，[解释一个 knn 模型](https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/kernel_explainer/Census%20income%20classification%20with%20scikit-learn.html)


------------

参考资料：
- 官方文档：https://shap-lrjball.readthedocs.io/en/latest/index.html
- [用 SHAP 可视化解释机器学习模型实用指南(上)](https://mp.weixin.qq.com/s?__biz=Mzk0OTI1OTQ2MQ==&mid=2247500066&idx=1&sn=fe878ccbbd1299366ada3ec9f622a402&chksm=c3599c88f42e159eef4da04751df3ed93aa3a0d53ad4d07c1a06036a9cd0bbb85c011afaa82d&scene=21#wechat_redirect)
- [用 SHAP 可视化解释机器学习模型实用指南(下)](https://cloud.tencent.com/developer/article/1888981)
- chatgpt