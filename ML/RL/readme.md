


强化学习，一种基于(智能体-环境-奖励)交互的机器学习算法，与监督学习是不同的范式，强化学习没有标签数据，只有延迟和稀疏的奖励反馈。

RL 更像是成长过程（不断进行尝试，在尝试中接受正向的奖励，或负向的评价，来优化决策水平），而不是利用已有智能（标签等）直接捏出一个机器人。（试错性，动态性）

RL 不仅是算法问题，更是系统设计问题。构造环境，奖励，在实际应用中非常关键。


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







一个完整的过程，通常叫 episod，整个生命周期的奖励 $R = \sum_{t=1}^Tr_t$


- on plicy，训练数据由当前 agent 不断环境交互得到
- off plicy，找个代练









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