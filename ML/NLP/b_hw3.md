> 大三下学期，自然语言处理，作业3：利用平均感知机实现词性标注

## 1. 实验要求

请大家完成利用感知机实现词性标注功能，要求如下（具体参考PPT）：
1. 跑通代码，完成模型训练，并给出实验报告；
2. 提交简要实验报告，写明自己做的工作并附上实验结果截图；
3. 实验报告格式：要求用Markdown(typora工具)完成实验报告，最终需要提交md后缀文件以及导出的pdf文件。


## 2. 基本原理

### 2.1 感知机

`直观定义`：感知机接受多个输入信号，输出一个信号，输入信号配以权重，用阈值 $\theta$ 判定这个神经元是否被激活。

`正规定义`：感知机是根据输入实例的特征向量 $x$ ，对其进行 **二分类** 的线性分类模型：
$$f(x)=sign(w \cdot x+b)$$


感知机算法是非常好的二分类算法，该算法求取一个分离超平面，超平面由w参数化并用来预测，对于一个样本 x，感知机算法通过计算 $y = [w,x]$ 预测样本的标签，最终的预测标签通过计算$sign(y)$来实现。算法仅在预测错误时修正权值w。

### 2.2 平均感知机

**平均感知机** 和感知机算法的训练方法一样，不同的是每次训练样本 $x_i$ 后，保留先前训练的权值，训练结束后平均所有权值。最终用平均权值作为最终判别准则的权值。参数平均化可以克服由于学习速率过大所引起的训练过程中出现的震荡现象。


## 3. 原始代码分析

### 3.1 文件结构

- data
  - `.pos`
    - 训练集 train
      - 训练算法
    - 开发集 dev
      - 调整参数、选择特征，以及对学习算法作出其它决定
    - 测试集 test
      - 开发集中选出的最优的模型在测试集上进行评估
- model
  - `.pkl`
- perc-tagger.ipynb
  - 读数据并训练
- percetron.py
  - 封装了 AveragedPerceptron 这个类

### 3.2 代码结构

```python
# 1：读取语料
# 读取 pos后缀的语料数据，生成 经过分词的 sentence 列表，和 词性列表 pos_tag
read_data()

# 2：特称提取
# 提取单词的字母、后缀，及其前后文信息
get_features()

# 3：进一步处理数据
# 生成可供 AveragedPerceptron 训练的数据
process_data()

# 4：normal 训练
# 原理：feats 作为感知机的输入， guess 作为结果
# 统计模型的训练评分
# 统计模型经过训练后，在 dev 数据集的评分

# 5：plus 训练
# 相对于 normal, word 在 single_pos_dict 中时，通过 dict 直接判断。

# 6：test
# 读取保存的模型，并判断评分
```

### 3.3 运行结果

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331164645.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331155753.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331164643.png">


## 4. 改进优化

### 4.1 提高训练次数

将 epoch 提高到 20 之后：

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331192229.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331192230.png">

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331185040.png">


相对于原先的 10 次 epoch，可见模型的准确度在不断提高

### 4.2 修改特征权值

修改 update 中特征权值大小，将 `+1` `-1` 分别改为
`weights.get(truth, 0.0)*0.1` `-weights.get(truth, 0.0)*0.1`。

结果如下：

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331203710.png">

### 4.3 加大罚项

将 `upd_feat(guess, f, weights.get(guess, 0.0), -1` 中的 -1 改为 -5。

这样模型在词性标注出错错时，会对出错的情况惩罚加大。

结果如下：

<img src="https://img-1301102143.cos.ap-beijing.myqcloud.com/20220331203452.png">

参考资料：
- [200行Python代码实现感知机词性标注器](http://www.hankcs.com/nlp/averaged-perceptron-tagger.html)

