


`Markov Decision Process`

马尔可夫决策过程 MDP, 由 （S, A, P, R） 组成

State S, Action A, Transition (state 之间转移的概率)P, Reward R



-------------


`贝尔曼方程`

强化学习的核心数学工具，用于描述最优策略下的状态值（或动作值）关系。

核心思想：当前状态价值 = 即时奖励 + 折扣后未来状态的价值


-------------

在 Q-learning 中，我们使用最优贝尔曼方程来更新 Q 值：

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma max_{a'}Q(s',a') - Q(s,a)]$$