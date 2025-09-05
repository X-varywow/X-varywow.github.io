

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

样本之间的强相关性，导致从连续样本中学习效率低下；所以随机化样本

将数据分布从时间相关转变为近似独立


`（2）目标网络`

使用独立的网络（目标网络）计算 TD 目标

$$y = r + \gamma max_{a^{\prime}}Q(s^{\prime},a^{\prime};\theta^-)$$

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




## demo.2048



```python
import numpy as np
import random
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

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
        # 将棋盘转换为对数尺度并归一化;
        # 关键信息在于数字的相对大小关系，2048 与 4096 指数级关系难以学习，更新幅度不均匀
        # 归一化避免梯度爆炸/消失
        # TODO: 尝试改变状态表示方式
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
        else:
            reward = -2
        
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
    
    def is_valid_move(self, action):

        old_board = self.board.copy()
        reward = 0
        
        # 旋转棋盘使移动方向统一为向左移动
        # 0 左移； 1 上移； 2 右移； 3 下移
        rotated_board = np.rot90(self.board, action)
        
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
        new_borad = np.rot90(rotated_board, -action)

        if np.array_equal(old_board, new_borad):
            self.board = old_board
            return False
        self.board = old_board
        return True


# play game by self
# env = Game2048()

# while True:
#     action = input()
#     if action == 'q':
#         break
#     arr = ['a', 'w', 'd', 's']
#     if action not in arr:
#         env.move(arr.index(action), mode='play')




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
    
    # 这里输出 1*4 的tensor, 代表状态下每个 action 的 q 值

# Dueling DQN的最后一层
# self.value_stream = nn.Linear(512, 1)  # 状态价值V(s)
# self.advantage_stream = nn.Linear(512, num_actions)  # 优势函数A(s,a)

# def forward(self, x):
#     x = F.relu(self.fc1(x))
#     V = self.value_stream(x)
#     A = self.advantage_stream(x)
#     return V + (A - A.mean())  # Q = V + (A - mean(A))




class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_shape, num_actions).to(self.device)
        self.target_model = DQN(state_shape, num_actions).to(self.device)

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
        
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # 添加batch和channel维度
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, episodes, tot_episodes):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)//2))
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).unsqueeze(1).to(self.device) # 添加 channel 维度
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)
        
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
            self.epsilon *= self.epsilon_decay
            # 总是 1 衰减到 (left + 1/e)
            # self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-1. * episodes / tot_episodes)
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()



def train(model_path, episodes=300):
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

        if e % 100 == 0:
            agent.update_target_model()
            print(f"Episode: {e}/{episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")
            print(env.board)
    agent.save(model_path)



def evaluate_model(model_path, num_games=10, render=False):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化环境和模型
    env = Game2048()
    model = DQN(input_shape=(1, 4, 4), num_actions=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    
    # 统计结果
    total_scores = []
    max_tiles = defaultdict(int)
    
    # 进行多场游戏测试
    for game in tqdm(range(num_games), desc="Testing"):
        state = env.reset()
        total_reward = 0
        current_max = 2
        
        while True:
            # 准备输入数据
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            
            # 模型预测
            with torch.no_grad():
                q_values = model(state_tensor)  # 假设形状为 [1, num_actions]
                
                # 创建有效动作掩码
                valid_mask = torch.tensor([[env.is_valid_move(i) for i in range(q_values.size(1))]], 
                                        dtype=torch.bool,
                                        device=q_values.device)
                
                # 生成处理后的Q值张量
                used_q = q_values.clone()  # 复制原始Q值
                used_q[~valid_mask] = -1e9  # 将无效动作的Q值设为极小值
                
                # 选择有效动作中Q值最大的动作
                action = used_q.argmax(dim=1).item()
            
            # 执行动作
            # 没有合并的块时，状态不转移，一直开在这里
            next_state, reward, done = env.move(action)
            total_reward += reward
            
            # 更新最大方块
            current_max = max(current_max, np.max(env.board))
            
            if render:
                print(f"Step reward: {reward:.1f}, Total: {total_reward:.1f}")
                print(env.board)
            
            if done:
                break
                
            state = next_state
            # print(env.board)
        
        # 记录结果
        total_scores.append(total_reward)
        max_tiles[current_max] += 1
        # print(env.board)
    
    # 计算统计数据
    avg_score = np.mean(total_scores)
    std_score = np.std(total_scores)
    max_score = np.max(total_scores)
    
    # 打印结果
    print("\n=== Evaluation Results ===")
    print(f"Games played: {num_games}")
    print(f"Average score: {avg_score:.1f} ± {std_score:.1f}")
    print(f"Highest score: {max_score:.1f}")
    print("\nMax tile distribution:")
    for tile in sorted(max_tiles.keys()):
        print(f"{tile}: {max_tiles[tile]} games ({max_tiles[tile]/num_games*100:.1f}%)")
    
    return avg_score, max_tiles


if __name__ == "__main__":
    model_path = "dqn_2048.pth"
    # v0, "./ZZZ/dqn_2048.pth";; 709.5 ± 411.2;;; 2716.0
    # v1, "./ZZZ/dqn_2048_v1.pth";; 初始版本, 500 次；；； 2496.8 ± 1229.6；； 6884.0
    # v2, "./ZZZ/dqn_2048_v2.pth"，  reward 无效动作 -2；；； 1149.7 ± 520.1； 2596.0
    # v2 500次, "./ZZZ/dqn_2048_v2.pth"，  reward 无效动作 -2；；； 2160.8 ± 1044.5； 5176.0


    # 调整衰减公式， 0.995；； 735.8 ± 409.4； 1900.0
    # 调整衰减公式， 1000， 0.995；； 824.4 ± 524.9； 1900.0

    # 有问题，训练得少反而效果更好

    train(model_path, 500)
    # evaluate_model(model_path, num_games=100, render=False)
```


## 总结


手动随便 5k 分； 出现2048，至少 2048 x (11-1) 分，ai 要 2w 分才合格，现在才 1000 分，，，

2048 许多关键策略（保持角落稳定，避免早期乱合并）需要长远规划，而 DQN 容易只学到局部贪心策略。所以效果不太好

atari 里的游戏多数是视觉反应型任务，如避开障碍，击打目标，局部目标技能指导动作；

gpt 推荐： 2048 使用启发式规则 + 搜索/强化学习


--------


基本只能局部最优，

对于离散空间和连续空间，分别采取什么策略？

action critic 两个网络，行动价值，状态价值；

> 这些东西基本都是通的，都是全局和局部的权衡，经济学中的机会成本，RL 中的状态价值。




----------

参考资料：
- https://zhuanlan.zhihu.com/p/441314394