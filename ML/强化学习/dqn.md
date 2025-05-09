

## preface

由 deepmind 2013 年提出，是 Q-learning 的扩展

**使用深度神经网络作为函数逼近器来估计 Q 值**，使其适用于高维或连续的状态和动作空间，而不是（Q-learning）在表格中存储Q值（不需要函数逼近）.


传统 Q-learning 具有以下问题：维度灾难，泛化能力差，无法处理连续状态；

DQN 重要意义：将深度学习强大表征能力引入到强化学习中


--------

关键技术：

`（1）经验回放`

存储智能体的经验（状态、动作、奖励、下一状态）到一个固定大小的缓冲区（replay buffer）

训练时随机采样一批历史经验（mini-batch），来消除序列数据中的事件依赖性和局部相关性，从而提升学习的稳定性和效率;

将数据分布从时间相关转变为近似独立


`（2）目标网络`

使用独立的网络（目标网络）计算 TD 目标

$$y = r + \gamma max_{a^{\prime}}Q(s^{\prime},a^{\prime};\theta^-)$$

目标网络参数定期更新（例如每隔若干步从主网络复制），避免目标值频繁波动






</br>

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

**步骤小结**：
- 初始化 Q 表，记录 (s,a) 的奖励值，大小：状态空间*每个空间的 action数
- 初始化动作空间，学习率 alpha、折扣因子 gamma(权衡当前奖励与未来奖励)、探索概率 epsilon
- 多轮训练，每轮从初始状态开始，循环（根据当前策略选择动作，执行动作，观察奖励和新状态，更新 Q 表）
- 多次训练共用一张 Q 表，所以对于未来的奖励能够在训练中影响前置的动作

> 其实传统 Q-learning 思路还算简单，核心在使用 贝尔曼方程维护Q表

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma max_{a'}Q(s',a') - Q(s,a)]$$

$$Q(s,a)\ \mathrel{+}=\ \alpha[r + \gamma max_{a'}Q(s',a') - Q(s,a)]$$

```
Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
```

$max_{a'}Q(s',a')$  新状态$s'$下所有可能动作最大 Q 值（最优未来回报），

这样先用着，贝尔曼详细推导有些麻烦



</br>

## demo.2048


### part1. 2048

```python
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()
        print(self.board)
    
    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.get_state()
    
    def add_random_tile(self):
        """在空白位置随机添加一个2或4"""
        empty_cells = [(i, j) for i in range(self.size) 
                      for j in range(self.size) if self.board[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 2 if random.random() < 0.9 else 4
    
    def get_state(self):
        """获取当前游戏状态"""
        # 将棋盘转换为对数尺度并归一化
        log_board = np.log2(self.board + 1)  # +1 避免log(0)
        return log_board / np.max(log_board) if np.max(log_board) > 0 else log_board
    
    def move(self, direction, mode = 'train'):
        """
        移动方块
        返回: (新状态, 获得的分数, 游戏是否结束)
        """
        old_board = self.board.copy()
        reward = 0
        
        # 旋转棋盘使移动方向统一为向左移动
        # 0 左移； 1 上移； 2 右移； 3 下移
        rotated_board = np.rot90(self.board, direction)
        
        # 移动和合并方块
        for i in range(self.size):
            row = rotated_board[i]
            row = row[row != 0]  # 移除0
            merged = []
            skip = False
            
            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    merged.append(row[j] * 2)
                    reward += row[j] * 2
                    skip = True
                else:
                    merged.append(row[j])
            
            # 填充0
            merged = merged + [0] * (self.size - len(merged))
            rotated_board[i] = merged
        
        # 旋转回原方向
        self.board = np.rot90(rotated_board, -direction)
        self.score += reward
        
        # 检查是否移动有效
        if not np.array_equal(old_board, self.board):
            self.add_random_tile()
        
        done = self.is_game_over()
        if mode == 'train':
            return self.get_state(), reward, done
        else:
            print(self.board)
            print(f"SCORE: {self.score}")
    
    def is_game_over(self):
        """检查游戏是否结束"""
        # 检查是否有空格
        if 0 in self.board:
            return False
        
        # 检查是否有可合并的相邻方块
        for i in range(self.size):
            for j in range(self.size):
                if j + 1 < self.size and self.board[i, j] == self.board[i, j + 1]:
                    return False
                if i + 1 < self.size and self.board[i, j] == self.board[i + 1, j]:
                    return False
        
        return True
```


get_state() 中 将棋盘转换为对数尺度并归一化，原因：
- 关键信息在于数字的相对大小关系，2048 与 4096 指数级关系难以学习，更新幅度不均匀
- 归一化避免梯度爆炸/消失

后续，可以试一下数据变换


-------------


```python
env = Game2048()

while True:
    action = input()
    if action == 'q':
        break
    arr = ['a', 'w', 'd', 's']
    if action not in arr:
        env.move(arr.index(action), mode='play')
```

随便 5k 分， 出现2048，至少 2048 x (11-1) 分，ai 要 2w 分才合格，现在才 1000 分，，，






### part2. dqn

```python
# 这里需要对应调整 batch_size 128?
# class DQN(nn.Module):
#     def __init__(self, n_actions):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(128 * 4 * 4, 256)
#         self.fc2 = nn.Linear(256, n_actions)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)  # 展平
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_shape[1] * input_shape[2], 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        
        # 这里输出 1*4 的tensor, 代表状态下每个 action 的 q 值
        return self.fc2(x)


class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.batch_size = 64
        self.model = DQN(state_shape, num_actions)
        self.target_model = DQN(state_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # 这个总是根据 Q 值选择动作，Q 值由 DQN 模型得出
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, episodes, tot_episodes):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).unsqueeze(1)  # 添加channel维度
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).unsqueeze(1)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        # 当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-1. * episodes / tot_episodes)
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()
```


### part3. train

```python
episodes = 1000
env = Game2048()
state_shape = (1, env.size, env.size)  # (channels, height, width)
agent = DQNAgent(state_shape, num_actions=4)

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.move(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    agent.replay(e, episodes)
    
    if e % 10 == 0:
        agent.update_target_model()
        print(f"Episode: {e}/{episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

agent.save("dqn_2048.pth")
```






--------


基本只能局部最优，

对于离散空间和连续空间，分别采取什么策略？

action critic 两个网络，行动价值，状态价值；

> 这些东西基本都是通的，都是全局和局部的权衡，经济学中的机会成本，RL 中的状态价值。




----------

参考资料：
- https://zhuanlan.zhihu.com/p/441314394