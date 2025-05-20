


`Markov Decision Process`

马尔可夫决策过程 MDP, 由 （S, A, P, R） 组成

State S, Action A, Transition (state 之间转移的概率)P, Reward R



-------------


`贝尔曼方程`

强化学习的核心数学工具，用于描述最优策略下的状态值（或动作值）关系。

核心思想：当前状态价值 = 即时奖励 + 折扣后未来状态的价值


状态价值函数 V(s) 表示从状态 s 开始，遵循策略 $\pi$ 的期望回报：

$$V^{\pi}(s) = \Epsilon_\pi[G_t | S_t = s]$$

其中累积回报 $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$, $\gamma$ 是折扣因子，$\gamma \in [0, 1]$。


$$V^{\pi}(s) = \sum_a\pi(a|s) \sum_{s',r}p(s',r|s,a)[r + \gamma V^{\pi}(s')]$$

- $\pi(a|s)$ 表示在状态 s 下选择动作 a 的概率
- $p(s',r|s,a)$ 表示在状态 s 下选择动作 a 后转移到状态 s' 并获得奖励 r 的概率


-------------

在 Q-learning 中，我们使用最优贝尔曼方程来更新 Q 值：

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma max_{a'}Q(s',a') - Q(s,a)]$$

$r + \gamma max_{a'}Q(s',a')$ 称之为 时序差分误差（TD Error）







</br></br></br>

-------------

参考资料：
- [知乎-贝尔曼方程](https://zhuanlan.zhihu.com/p/688029400)
- gpt