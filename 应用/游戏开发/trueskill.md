

基于贝叶斯推断的评分系统，微软开发用以替代 elo 评分系统

https://trueskill.org/


</br>

## _基础组件_


贝叶斯网络（概率统计的应用、进行不确定性推理和数值分析的工具）

<img src="http://math.itdiffer.com/images/bayesnetwork01.png" style="zoom:50%">

图中每个点表示一个特征，那么：
- 无关（独立）， A and C, B and C
- 相关, 如 A -> B , 这个关系用条件概率表示为 P(B|A), 类似有 P(D|B,C)
- 条件无关（条件独立），D 认为有条件地独立于 A，P(D|B, A) = P(D|B)

上述图，称为 概率图模型 （Probabilistic Graphical Model, PGM）

在概率图模型中，每个结点表示一个随机变量，结点之间的边表示各随机变量的依赖关系。所以，图表示联合概率分布






因子图

- 贝叶斯网络中的一个因子（函数）对应因子图中的一个结点
- 将边际条件概率表示为变量
- 因子节点向其他节点发送信息，这些消息有助于简化计算












</br>

## _trueskill_


主要迭代过程：

$$\mu_w \leftarrow \mu_w + \frac{\sigma_w^2 }{c_{ij}}\cdot v(\frac{\mu_w - \mu_l}{c_{ij}},\frac{\epsilon}{c_{ij}}) \quad and \quad \sigma_w \leftarrow  \sigma_w\sqrt{1-\frac{\sigma_w^2}{c_{ij}^2} \cdot w (\frac{\mu_w - \mu_l}{c_{ij}},\frac{\epsilon}{c_{ij}})}$$

$$\mu_l \leftarrow \mu_l - \frac{\sigma_l^2 }{c_{ij}}\cdot v(\frac{\mu_w - \mu_l}{c_{ij}},\frac{\epsilon}{c_{ij}}) \quad and \quad \sigma_l \leftarrow  \sigma_l\sqrt{1-\frac{\sigma_l^2}{c_{ij}^2} \cdot w (\frac{\mu_w - \mu_l}{c_{ij}},\frac{\epsilon}{c_{ij}})}$$

-----

说明：

W 表示胜，L 表示输

$$c_{ij}^2 = 2\beta^2 + \sigma_w^2 + \sigma_l^2$$

$$v(t,\alpha) := \frac{\Nu(t-\alpha;0,1)}{\Phi(t-\alpha)}$$

$$w(t, \alpha) := v(t, \alpha)\cdot (v(t, \alpha) + (t-\alpha))$$






</br>

## _TTT_

Trueskill through time

代码仓库：https://github.com/glandfried/TrueSkillThroughTime.py

----------

原始trueskill不足：

推断结果依赖于 输入数据的顺序

无法做到信息传递（如A 击败了 B,  但是 B 后续表现上看，已经完全脱离了原本水平）


---------

解决方法：

extending the Gaussian density filtering to running full expectation propagation (EP) until convergence 
将高斯密度滤波扩展到完全期望传播（EP）直到收敛   `？？？`

---------

步骤：

记 $ p_{ij}^t(k)$ 为玩家发挥表现，$s_1$ 和 $s_2$ 为玩家能力，则 $p_1 \sim N(s_1, \beta^2)$

记 $y_{ij}^t(k)$ 为比赛结果，i, j 为 player, t 为 time, $\epsilon$用来判断平局,


$$y_{ij}^t(k) = \begin{cases}
+1, \quad if \quad p_{ij}^t(k) > p_{ji}^t(k) + \epsilon\\
-1, \quad if \quad p_{ji}^t(k) > p_{ij}^t(k) + \epsilon\\
0
 \end{cases}$$

记 $m_{f(p_{ij}^t(k), s_i^t) \rightarrow s_i^t} (s_i^t)$ 为在 $s_i^t$ 更新后的结果 $p_{ij}^t(k)$

新的下行信息计算如下：

$$m_{f(p_{ij}^t(k), s_i^t) \rightarrow p_{ij}^t(k)}(p_{ij}^t(k)) = 。。。$$

然后根据 t 反复计算下行信息

$$m_{f(s_i^{t-1},s_i^t)\rightarrow s_i^{t-1}}(s_i^{t-1}) = \int_{-\infty}^{\infty}f(s_i^{t-1},s_i^t)\frac{p(s_i^t)}{m_{f(s_i^{t-1},s_i^t)\rightarrow s_i^{t}}(s_i^{t})}ds_i^t$$

根据时序计算到收敛，


</br>

## _trueskill2_

trueskill 提供了一个 动态 K 因子，使用贝叶斯推断，相对 elo 几个改进：
- 收敛较快
- 原生支持团队
- 更合理的平局处理




----------

参考资料：
- https://trueskillthroughtime.readthedocs.io/en/latest/
- [Trueskill 原理简介](https://zhuanlan.zhihu.com/p/48737998)
- [《TureSkill2评分机制详解》](https://zhuanlan.zhihu.com/p/568689092) ⭐️
- [Trueskill原理与应用（ppt）](https://zhuanlan.zhihu.com/p/560942120)