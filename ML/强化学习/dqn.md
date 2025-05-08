

很有意思啊

## preface

由 deepmind 2013 年提出，是 Q-learning 的扩展

**使用深度神经网络作为函数逼近器来估计 Q 值**，使其适用于高维或连续的状态和动作空间，而不是（Q-learning）在表格中存储Q值（不需要函数逼近）.


传统 Q-learning 具有以下问题：维度灾难，泛化能力差，无法处理连续状态；

DQN 重要意义：将深度学习强大表征能力引入到强化学习中


--------

关键技术：

`（1）经验回放`

存储智能体的经验（状态、动作、奖励、下一状态）到一个固定大小的缓冲区（replay buffer）

训练时随机采样一批历史经验（mini-batch），打破数据间的相关性，提高稳定性


`（2）目标网络`

使用独立的网络（目标网络）计算 TD 目标

$$y = r + \gamma max_{x^{\prime}}Q(s^{\prime},a^{\prime};\theta^-)$$

目标网络参数定期更新（例如每隔若干步从主网络复制），避免目标值频繁波动








## Q-learning


demo. 寻路


```python
import numpy as np
import random

# 定义环境
grid = [
    ['S', ' ', ' ', ' ', ' '],
    [' ', 'X', ' ', 'X', ' '],
    [' ', ' ', ' ', 'X', ' '],
    [' ', 'X', ' ', ' ', ' '],
    [' ', ' ', ' ', 'X', 'G']
]

# 参数
n_states = 5 * 5  # 5x5 的格子
n_actions = 4      # 上(0)、下(1)、左(2)、右(3)
alpha = 0.1        # 学习率
gamma = 0.9        # 折扣因子
epsilon = 0.1      # 探索概率
episodes = 1000    # 训练轮次

# 初始化 Q 表
Q = np.zeros((n_states, n_actions))

# 状态编码：将 (x, y) 坐标转换为唯一状态编号
def state_to_idx(x, y):
    return x * 5 + y

# 动作定义
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右

# 训练 Q-learning
for episode in range(episodes):
    # 初始化起点 (0, 0)
    x, y = 0, 0
    state = state_to_idx(x, y)
    
    while True:
        # ε-greedy 选择动作
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)  # 随机探索
        else:
            action = np.argmax(Q[state])  # 选择当前最优动作
        
        # 执行动作
        dx, dy = actions[action]
        new_x, new_y = x + dx, y + dy
        
        # 边界检查
        if new_x < 0 or new_x >= 5 or new_y < 0 or new_y >= 5:
            new_x, new_y = x, y  # 保持原位
        
        # 检查是否碰到障碍物
        if grid[new_x][new_y] == 'X':
            new_x, new_y = x, y  # 保持原位
        
        new_state = state_to_idx(new_x, new_y)
        
        # 定义奖励
        if grid[new_x][new_y] == 'G':
            reward = 10  # 到达目标
            done = True
        else:
            reward = -1  # 每步惩罚（鼓励尽快到达目标）
            done = False
        
        # 更新 Q 表
        Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
        
        # 更新状态
        x, y = new_x, new_y
        state = new_state
        
        if done:
            break

# 测试训练结果
x, y = 0, 0
path = [(x, y)]
while True:
    state = state_to_idx(x, y)
    action = np.argmax(Q[state])
    dx, dy = actions[action]
    x, y = x + dx, y + dy
    path.append((x, y))
    
    if grid[x][y] == 'G':
        print("找到目标！路径：", path)
        break
    if len(path) > 20:  # 防止无限循环
        print("路径过长，可能未收敛。")
        break
```

步骤小结：
- 初始化 Q 表，记录 (s,a) 的奖励值，大小：状态空间*每个空间的 action数
- 初始化动作空间，学习率 alpha、折扣因子 gamma(权衡当前奖励与未来奖励)、探索概率 epsilon
- 多轮训练，每轮从初始状态开始，循环（根据当前策略选择动作，执行动作，观察奖励和新状态，更新 Q 表）


> 其实传统 Q-learning 思路还算简单，核心在使用 贝尔曼方程维护Q表





--------


基本只能局部最优，

对于离散空间和连续空间，分别采取什么策略？

action critic 两个网络，行动价值，状态价值；

> 这些东西基本都是通的，都是全局和局部的权衡，经济学中的机会成本，RL 中的状态价值。




----------

参考资料：
- https://zhuanlan.zhihu.com/p/441314394