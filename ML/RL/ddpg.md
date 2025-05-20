Deterministic Policy Gradient, 适用于 连续动作空间，使用 actor-critic 架构。

-----------

D4PG 是 DeepMind 提出的一个强化学习算法，属于 离策略、连续动作空间 的强化学习方法, 在 DDPG 基础上改进而来。

- 更稳定、收敛更快：分布式 Q 值和 n-step return 带来更丰富的训练信号。
- 更高的数据利用效率：并行采样 + 离策略学习。
- 适用于高维连续控制问题：如 MuJoCo（一个物理模拟引擎）、机器人控制等。
