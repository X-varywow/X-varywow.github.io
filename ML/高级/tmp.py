import math
import random


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state          # 当前节点的状态
        self.parent = parent        # 父节点
        self.action = action        # 从父节点到该节点的动作
        self.children = []          # 子节点
        self.visits = 0             # 节点被访问的次数
        self.value = 0.0            # 节点的累计奖励

    def is_leaf(self):
        # 判断当前节点是否已完全扩展
        if len(self.children) >= len(self.state.get_legal_actions()):
            return True
        return False

    def best_child(self, exploration_weight=1.0):
        """基于 UCB1（Upper Confidence Bound）选择最优子节点"""
        if not self.children:
            raise Exception("No children to select from")

        def ucb1(node):
            return (node.value / (node.visits + 1e-6) +
                    exploration_weight * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6)))

        return max(self.children, key=ucb1)


class MCTS:
    def __init__(self, exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, num_simulations):
        root = Node(initial_state)

        for _ in range(num_simulations):
            node = self._select(root)
            if not node.state.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        return root.best_child(exploration_weight=0).state

    def _select(self, node):
        """选择最优路径中的叶节点"""
        while not node.state.is_terminal() and node.is_leaf():
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node):
        """扩展一个子节点"""
        actions = node.state.get_legal_actions()
        tried_actions = [child.action for child in node.children]
        for action in actions:
            if action not in tried_actions:
                next_state = node.state.perform_action(action)
                child_node = Node(next_state, parent=node, action=action)
                node.children.append(child_node)
                return child_node

        raise Exception("Should never reach here")

    def _simulate(self, state):
        """随机模拟直到终局并返回奖励"""
        while not state.is_terminal():
            action = random.choice(state.get_legal_actions())
            state = state.perform_action(action)
        return state.get_reward()

    def _backpropagate(self, node, reward):
        """回溯更新节点统计信息"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class TicTacToeState:
    def __init__(self, board=None, current_player=1):
        # 初始化棋盘
        self.board = board or [0] * 9  # 0 表示空格，1 表示玩家1，-1 表示玩家2
        self.current_player = current_player

    def get_legal_actions(self):
        # 返回所有可用的动作（棋盘上为空的位置）
        return [i for i in range(9) if self.board[i] == 0]

    def perform_action(self, action):
        # 执行动作，返回新的状态
        new_board = self.board[:]
        new_board[action] = self.current_player
        return TicTacToeState(new_board, -self.current_player)

    def is_terminal(self):
        # 判断是否为终止状态
        return self.get_winner() is not None or all(cell != 0 for cell in self.board)

    def get_winner(self):
        # 检查是否有玩家获胜
        winning_positions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 横排
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 竖列
            (0, 4, 8), (2, 4, 6)              # 对角线
        ]
        for pos in winning_positions:
            line = [self.board[i] for i in pos]
            if abs(sum(line)) == 3:  # 如果某条线的总和为3或-3，则有玩家获胜
                return line[0]
        return None

    def get_reward(self):
        # 返回奖励：1 表示玩家1胜利，-1 表示玩家2胜利，0 表示平局
        winner = self.get_winner()
        return winner if winner is not None else 0

    def __str__(self):
        # 返回棋盘的字符串表示
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [
            " ".join(symbols[self.board[i]] for i in range(j, j + 3))
            for j in range(0, 9, 3)
        ]
        return "\n".join(rows)


# 测试 MCTS 在井字棋中的表现
if __name__ == "__main__":
    # 初始化状态
    initial_state = TicTacToeState()
    mcts = MCTS()

    print("Initial Board:")
    print(initial_state)

    while not initial_state.is_terminal():
        if initial_state.current_player == 1:
            # 使用 MCTS 选择最佳动作
            print("\nPlayer 1 (X) is thinking...")
            initial_state = mcts.search(initial_state, num_simulations=1000)
        else:
            # 模拟玩家 2 随机移动
            print("\nPlayer 2 (O) is thinking...")
            action = random.choice(initial_state.get_legal_actions())
            initial_state = initial_state.perform_action(action)

        print("\nBoard after move:")
        print(initial_state)

    # 游戏结束，显示结果
    winner = initial_state.get_winner()
    if winner == 1:
        print("\nPlayer 1 (X) wins!")
    elif winner == -1:
        print("\nPlayer 2 (O) wins!")
    else:
        print("\nIt's a draw!")
