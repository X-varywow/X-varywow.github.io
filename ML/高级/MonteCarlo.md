

## _Preface_


先验概率是 以全事件为背景下,A事件发生的概率，P(A|Ω)

后验概率是 以新事件B为背景下,A事件发生的概率， P(A|B)



全事件一般是统计获得的，所以称为 <u>先验概率，没有实验前的概率</u>

新事件一般是实验，如试验B，此时的事件背景从全事件变成了B，该事件B可能对A的概率有影响，那么需要对A现在的概率进行一个修正，从P(A|Ω)变成 P(A|B)，

所以称 P(A|B)为后验概率，也就是试验(事件B发生)后的概率


>结果发生在事件之前，就是先验的。发生在事件（实验）之后，就是后验的。简单些：先验是先前的经验，后验是后来的经验。




</br>

## _Monte Carlo_

蒙特卡洛：一种统计模拟的方法，用大量模拟发生的现象来测算研究对象。（听着挺厉害，实际就是个大量重复模拟）

eg. 选定矩形，随机选点，可求复杂函数的定积分


----------

蒙特卡洛搜索树：使用博弈树来表示一个游戏，博弈树每个节点都代表一种状态；当状态多到无法穷举的时候，大数定律（采样数量足够大，采样样本可以无限近似地表示原分布），蒙特卡洛树搜索是一种使用随机采样来近似估计的方法，通过大量自博弈来寻找“最有可能走”的节点；每次选择 UCT 值最高的节点访问。

$$UCT = \frac{w_i}{n_i} + C \sqrt{\frac{ln(N)}{n_i}}$$

- $w_i$ 节点 i 的累计奖励 （除下来就是平均奖励，合理的）
- $n_i$ 节点 i 被访问的次数
- $N$   当前节点总访问次数（在遍历子节点进行选择时，即为父节点的）
- $c$   探索参数，越高越倾向于探索新节点；常用默认值 $\sqrt2$


跟 Upper Confidence Bound 1 `UCB1` 一样的？在探索与复用之间平衡的，拍出来的一个式子




</br>

## _MCTS_


Monte Carlo Tree Search, 在状态树中进行搜索，逐步优化策略，可适用于状态空间巨大的问题。



| 主要步骤      | 说明                                                                          |
| ------------- | ----------------------------------------------------------------------------- |
| select        | 在当前节点上，将子节点的 value & visits 带入 uct 得出概率组，选择最大的       |
| expand        | 在当前节点上，扩展一个尚未探索（未在 children 中）的动作，**这里可插入价值网络** |
| simulate      | 选定节点随机进行模拟直到游戏结束                                              |
| backpropagate | 得到模拟结果后，往前更新路径节点的参数（visits, value）                       |


流程伪代码：
```python
for iter:
    root.select 走到叶子结点
    if 正常：
        new_node = node.expand()
        if new_node:
            reward = new_node.simulate()
            new_node.backpropagate()
```

----------

demo. mcts 井字棋：

```python
import random
import math


class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 井字棋盘
        self.current_player = 'X'

    def get_available_moves(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, move):
        new_state = TicTacToe()
        new_state.board = self.board[:]
        new_state.board[move] = self.current_player
        new_state.current_player = 'O' if self.current_player == 'X' else 'X'
        return new_state

    def check_winner(self):
        win_patterns = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for (i, j, k) in win_patterns:
            if self.board[i] == self.board[j] == self.board[k] and self.board[i] != ' ':
                return self.board[i]  # 返回胜者 ('X' 或 'O')
        return None if ' ' in self.board else 'Draw'  # 平局或未结束


class Node:
    def __init__(self, state, parent=None):
        self.state = state  # 游戏状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点
        self.visits = 0   # 访问次数
        self.wins = 0     # 胜利次数

    # is_leaf     
    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_available_moves())

    def best_child(self, exploration_weight=1.41):
        return max(self.children, key=lambda child: (child.wins / (child.visits + 1e-6)) + 
                   exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)))


class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def search(self, initial_state):
        root = Node(initial_state)
        for _ in range(self.iterations):
            node = self._select(root)
            winner = self.a(node.state)
            self._backpropagate(node, winner)
        return root.best_child(exploration_weight=0).state

    def _select(self, node):
        """选择最优路径中的叶节点"""
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _expand(self, node):
        """扩展一个子节点"""
        available_moves = node.state.get_available_moves()
        tried_moves = {tuple(child.state.board) for child in node.children}  # 使用 tuple 作为哈希
        for move in available_moves:
            new_state = node.state.make_move(move)
            if tuple(new_state.board) not in tried_moves:
                child = Node(new_state, parent=node)
                node.children.append(child)
                return child
        return node


    def _simulate(self, state):
        """随机模拟直到终局并返回奖励"""
        current_state = state
        while True:
            winner = current_state.check_winner()
            if winner:
                return winner
            move = random.choice(current_state.get_available_moves())
            current_state = current_state.make_move(move)

    def _backpropagate(self, node, winner):
        """回溯更新节点统计信息"""
        while node:
            node.visits += 1
            if winner == node.state.current_player:  # 反向奖励
                node.wins += 1
            node = node.parent

# 测试 MCTS 井字棋
if __name__ == "__main__":
    game = TicTacToe()
    mcts = MCTS(iterations=1000)
    while True:
        if game.current_player == 'X':  # 让 MCTS 作为 X 玩
            game = mcts.search(game)
        else:
            move = int(input("Enter your move (0-8): "))
            game = game.make_move(move)
        print("\nBoard:")
        print(game.board[:3])
        print(game.board[3:6])
        print(game.board[6:])
        winner = game.check_winner()
        if winner:
            print("Winner:", winner)
            break

```





</br>

## _Alpha0_

> 以一个项目为例子看看
> - https://github.com/junxiaosong/AlphaZero_Gomoku
> - https://zhuanlan.zhihu.com/p/32089487


AlphaZero_Gomoku 使用 mcts_pure 和 mcts_alpha0 对弈获取训练数据，

训练更新 mcts_alpha0 的策略价值模型（PolicyValueNet），在单次模拟动作的时候用到;

记录每个格子的空位 availables，用于 get_moves


---------


单步模拟动作：先 UCB 走状态树（select 操作，根据 ucb 平衡探索与奖励计算出 概率，然后选取最大的child）走到叶子结点，根据 policy net 确定下一步，模拟出结果回溯更新所有父节点;





</br>

mcts_pure 和 mcts_alpha0 不同处：
- mcts_pure 使用默认策略价值函数，等概率




-------------

参考资料：
- https://www.bilibili.com/video/BV1hV4y1Q7TR/
- [如何理解先验概率与后验概率](https://zhuanlan.zhihu.com/p/26464206)评论区
- chatgpt