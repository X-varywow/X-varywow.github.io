

## 基本原理


`SVM` 是一种二类分类模型

- 基本模型是定义在特征空间上的 **间隔最大** 的线性分类器
- 有别于感知机，间隔最大化使其解唯一
- 实质上的非线性分类器, 支持非线性的核函数

----------------------

- 学习策略：`最大间隔法`，可以表示为凸二次规划问题。其原始最优化问题：

$$\min_{w,b}\frac12|w|^2$$

- 凸集
  - 在欧氏空间中，凸集是对于集合内的每一对点，连接该对点的直线段上的每个点也在该集合内。

---------------------

- **线性可分支持向量机**
  - 学习策略：硬间隔最大化
- **线性支持向量机**
  - 学习策略：软间隔最大化
- **非线性支持向量机**
  - 学习策略：核技巧 + 软间隔最大化
  - **核技巧**：通过空间映射，隐式地在高维的特征空间中学习线性支持向量机



## 题目要求

1. 了解SVM工具包sklearn.svm
   1. 从 start_code 开始
   2. 阅读文档
   3. 熟悉使用sklearn.svm.SVC（SVM分类器）接口
2. 练习使用SVM分类器，分别针对toy_data、random_data、iris_data，分别运行各种参数配置：
   1. 使用不同核函数（kernel）、不同罚项系数C，运行分类器查看分类效果；
   2. 将iris_data按一定比例（比如9:1）随机划分为训练集和测试集，实验获得最好效果的SVM超参数配置；
   3. 打印SVM分类器最终获得的支持向量集；在random_data数据集上使用matplotlib可视化分类效果：分类结果、支持向量点、分割面、margin边界


## 代码实现


##### start_code
```python
import numpy as np
import sklearn.svm as SVM
from sklearn.metrics import classification_report 


def iris_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target

def toy_data():
    X = np.array([[3, 3], [4, 3], [1, 1]])  
    Y = np.array([1, 1, -1])
    return X, Y  

def random_data():
    np.random.seed(0)
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    Y = [0] * 20 + [1] * 20
    return X, Y

def main():
    X, Y = toy_data()
    #X, Y = random_data()
    #X, Y = iris_data()
    model = SVM.SVC(kernel='linear').fit(X, Y)
    pred = model.predict(X)
    print(classification_report(Y, pred))
    print(Y)
    print(pred)
    print(Y == pred)

if __name__ == '__main__':
    main()
```

##### after
<a href="main/zone/svm.html" target="_blank">代码参考</a>


--------------

参考资料：
- [《统计机器学习》视频讲解](https://www.bilibili.com/video/BV1o5411p7H2)
- [如何理解拉格朗日乘子法](https://www.zhihu.com/question/38586401)
- [sklearn-svm 官方文档](https://scikit-learn.org/stable/modules/svm.html)
- [sklearn.svm.SVC 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- 《统计机器学习》