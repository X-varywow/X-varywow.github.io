


强化学习，原理：与 “绩效奖励” 类似

特性：与监督学习、非监督学习不同，RL如同一个婴儿的成长（不断进行尝试，在尝试中接受正向的奖励，或负向的评价，来优化婴儿的决策水平），而不是利用已有智能直接捏出一个机器人。


## _基本概念_

几个关键因素：主体，目标，环境，行动，奖励

</br>
折扣回报

$$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + ...$$


</br>
动作价值函数

$$Q_\pi (s_t, a_t) = E[U_t|S_t = s_t, A_t = a_t]$$


</br>
状态价值函数

$$V_\pi(s_t) = E_{A_t}[Q_\pi (s_t, A_t)]$$


</br>
策略梯度

$$\frac{\partial V_\pi(s_t)}{\partial \theta} = E_{A_t \sim \pi}[Q_\pi (s_t, A_t) \frac{\partial log(\pi(A_t|s_t;\theta))}{\partial \theta}]$$






</br>

## _A3C_

每个进程分别与环境进行交互学习；异步并行训练框架；

解决单个智能体与环境交互收集速度慢，训练难以收敛的问题；

[参考](https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9tb2RlbGFydHMtbGFicy1iajQtdjIub2JzLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vY291cnNlL21vZGVsYXJ0cy9yZWluZm9yY2VtZW50X2xlYXJuaW5nL3BvbmdfQTNDL1BvbmctQTNDLmlweW5i)


</br>

## _PPO_

proximal policy optimization, 近端策略优化。(OpenAI 2017)

PPO的主要特点是尝试保持新旧策略之间的差异适度，即它限制了策略更新过程中策略变化的幅度以避免过于激烈的调整导致性能下降。它通过截断的策略梯度或裁剪（clipping）约束来实现。

具体地，PPO定义了一个“剪切”目标函数，用来维持两个连续策略之间的一个预设界限，如果实际的策略变动超过这个界限，该目标函数会限制该变动，从而提高算法的稳定性。




</br>

## _DQN_

维护一张 Q 表

动作、状态太多时，



--------


基本只能局部最优，

对于离散空间和连续空间，分别采取什么策略？

action critic 两个网络，行动价值，状态价值；

> 这些东西基本都是通的，都是全局和局部的权衡，经济学中的机会成本，RL 中的状态价值。



</br>

## _逆向强化学习_

复杂场景特别是多智能体的博弈下，给出激励（Reward）是极其困难的，多数情况几乎不可行。

Inverse Reinforcement Learning 通过收集专家的经验与环境信息，来反向学习激励函数。

-----------

参考资料：
- https://easyai.tech/ai-definition/reinforcement-learning/
- [DQN打只狼里的boss](https://www.bilibili.com/video/BV1by4y1n7pe/)
- [强化学习—— 离散与连续动作空间](https://blog.csdn.net/Cyrus_May/article/details/124137445)
- https://mp.weixin.qq.com/s/NvwaR_dzQZnE85W1YtVWiA
- chatgpt