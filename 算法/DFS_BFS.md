## [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid: return 0
        rows, cols = len(grid), len(grid[0])
        res = 0

        def dfs(i, j):
            grid[i][j] = '0'
            for x, y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                if 0<=x<rows and 0<=y<cols and grid[x][y] == '1':
                    dfs(x, y)

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res += 1

        return res
```

## [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

给定一个包含了一些 `0` 和 `1` 的非空二维数组 `grid` 。

一个 岛屿 是由一些相邻的 `1` (代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在水平或者竖直方向上相邻。你可以假设 `grid` 的四个边缘都被 `0`（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 `0`)

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        res = 0

        def dfs(i, j):
            grid[i][j] = 0
            cnt = 1
            for x, y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                if 0<=x<rows and 0<=y<cols and grid[x][y]:
                    cnt += dfs(x, y)
            return cnt

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    res = max(res, dfs(i, j))
        return res

```

## [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

返回矩阵中最长递增路径的长度。

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        rows, cols = len(matrix), len(matrix[0])

        @lru_cache(None)
        def dfs(i, j):
            best = 1
            for x,y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                if 0<=x<rows and 0<=y<cols and matrix[x][y]>matrix[i][j]:
                    best = max(best, dfs(x, y) + 1)
            return best
        
        res = 0
        for i in range(rows):
            for j in range(cols):
                res = max(res, dfs(i,j))
        return res

# 记忆化 + DFS
```

## [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        rows, cols = len(mat), len(mat[0])
        q = collections.deque([])
        vis = set()

        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    q.append((i,j))
                    vis.add((i,j))

        while q:
            i, j = q.popleft()
            for x, y in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if 0<=x<rows and 0<=y<cols and (x,y) not in vis:
                    mat[x][y] = mat[i][j]+1
                    vis.add((x,y))
                    q.append((x,y))
        
        return mat
```


## [1219. 黄金矿工](https://leetcode.cn/problems/path-with-maximum-gold/)

你要开发一座金矿，地质勘测学家已经探明了这座金矿中的资源分布，并用大小为 `m * n` 的网格 `grid` 进行了标注。每个单元格中的整数就表示这一单元格中的黄金数量；如果该单元格是空的，那么就是 `0`。

为了使收益最大化，矿工需要按以下规则来开采黄金：

- 每当矿工进入一个单元，就会收集该单元格中的所有黄金。
- 矿工每次可以从当前位置向上下左右四个方向走。
- 每个单元格只能被开采（进入）一次。
- 不得开采（进入）黄金数目为 0 的单元格。
- 矿工可以从网格中 **任意一个** 有黄金的单元格出发或者是停止。

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:

        def dfs(i,j):
            if 0<=i<rows and 0<=j<cols and grid[i][j]:
                tmp,grid[i][j]= grid[i][j],0
                gain=max(dfs(i-1,j),dfs(i+1,j),dfs(i,j-1),dfs(i,j+1))
                grid[i][j]=tmp
                return tmp+gain
            else:
                return 0

        rows,cols=len(grid),len(grid[0])
        ans=0
        for i in range(rows):
            for j in range(cols):
                ans=max(ans,dfs(i,j))
        return ans

# dfs带上回溯
```

## [841. 钥匙和房间](https://leetcode.cn/problems/keys-and-rooms/)

有 `N` 个房间，开始时你位于 `0` 号房间。每个房间有不同的号码：`0，1，2，...，N-1`，并且房间里可能有一些钥匙能使你进入下一个房间。

在形式上，对于每个房间 `i` 都有一个钥匙列表 `rooms[i]`，每个钥匙 `rooms[i][j]` 由 `[0,1，...，N-1]` 中的一个整数表示，其中 `N = rooms.length`。 钥匙` rooms[i][j] = v` 可以打开编号为 `v` 的房间。

最初，除 `0` 号房间外的其余所有房间都被锁住。

你可以自由地在房间之间来回走动。

如果能进入每个房间返回 `true`，否则返回 `false`。

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited=set()

        def dfs(i):
            visited.add(i)
            for room in rooms[i]:
                if room not in visited:
                    dfs(room)
        
        dfs(0)

        return len(list(visited))==len(rooms)
```

[529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/)
----------------

```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        rows,cols=len(board),len(board[0])
        if board[click[0]][click[1]]=="M":
            board[click[0]][click[1]]="X"
            return board

        def check(x,y):
            cnt=0
            for i,j in[(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]:
                if 0<=i<rows and 0<=j<cols:
                    if board[i][j]=="M": cnt+=1
            return cnt

        def dfs(x,y):
            if x<0 or x>=rows or y<0 or y>=cols:
                return 
            if board[x][y]=="E":
                tmp=check(x,y)
                if tmp==0:
                    board[x][y]="B"
                    for i,j in[(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]:
                        dfs(i,j)
                else:
                    board[x][y]=str(tmp)

        dfs(click[0],click[1])
        return board

# 看起来抽象的东西不一定难
```

[1162.地图分析](https://leetcode-cn.com/problems/as-far-from-land-as-possible/)
-----------

```python
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        n=len(grid)
        steps=-1
        queue=[(i,j) for i in range(n) for j in range(n) if grid[i][j]==1]
        if len(queue) in [0,n*n]: return -1
        while len(queue)>0:
            for _ in range(len(queue)):
                x,y=queue.pop(0)
                for xi,yi in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                    if 0<=xi<n and 0<=yi<n and grid[xi][yi]==0:
                        queue.append((xi,yi))
                        grid[xi][yi]=-1
            steps+=1
        return steps

# BFS
# 从陆地（1）扩展到海洋（0）
```

[面试题 16.19. 水域大小](https://leetcode-cn.com/problems/pond-sizes-lcci/)
------------------

##### 1. DFS
```python
class Solution:
    def pondSizes(self, land: List[List[int]]) -> List[int]:
        rows, cols = len(land), len(land[0])
        ans=[]

        def dfs(r, c):
            if not(0<=r<rows and 0<=c<cols and land[r][c]==0): return 0
            land[r][c]=1
            return 1+dfs(r-1,c-1)+dfs(r,c-1)+dfs(r+1,c-1)+dfs(r-1,c)+dfs(r+1,c)+dfs(r-1,c+1)+dfs(r,c+1)+dfs(r+1,c+1)
        for r in range(rows):
            for c in range(cols):
                if land[r][c] == 0:
                    ans.append(dfs(r,c))

        return sorted(ans)
```

##### 2. BFS

```python
class Solution:
    def pondSizes(self, land: List[List[int]]) -> List[int]:

        rows,cols=len(land),len(land[0])
        ans=[]

        def bfs(i,j):
            queue=[(i,j)]
            size=0
            while len(queue)>0:
                x,y=queue.pop(0)
                size+=1
                for xi,yi in [(x+1,y),(x+1,y+1),(x+1,y-1),(x,y+1),(x,y-1),(x-1,y+1),(x-1,y),(x-1,y-1)]:
                    if 0<=xi<rows and 0<=yi<cols and land[xi][yi]==0:
                        queue.append((xi,yi))
                        land[xi][yi]=-1
            return size

        for i in range(rows):
            for j in range(cols):
                if land[i][j]==0:
                    land[i][j]=-1
                    ans.append(bfs(i,j))
        ans.sort()
        return ans
# return ans.sort() 返回的 null
```


[1042. 不邻接植花](https://leetcode-cn.com/problems/flower-planting-with-no-adjacent/)
---------------------------

> 图在 python 中，通常使用 defaultdict(list) 表示
```python
class Solution:
    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        d = defaultdict(list)
        for u,v in paths:
            d[u].append(v)
            d[v].append(u)

        res = [0] * n
        vis = set()

        def dfs(i):
            vis.add(i)
            c = [res[j-1] for j in d[i]]
            for use in [1,2,3,4]:
                if use not in c:
                    res[i-1] = use
                    break

            for u in d[i]:
                if u not in vis:
                    dfs(u)

        for i in range(n):
            if res[i]==0:
                dfs(i+1)
        
        return res
```

[剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)
---------------------

无重复字符串的排列组合。编写一种方法，计算某字符串的所有排列组合，字符串每个字符均不相同。

```python
# 方法一：用itertools库中的permutations
return [''.join(i) for i in list(permutations(S))]

#方法二：回溯
class Solution:
    def permutation(self, s: str) -> List[str]:
        if not s: return []
        ans = set()
        def dfs(s, path):
            if not s:
                ans.add(path)
                return
            for i in range(len(s)):
                dfs(s[:i]+s[i+1:], path+s[i])
        dfs(s, "")
        return list(ans)
```

[909.蛇梯棋](https://leetcode-cn.com/problems/snakes-and-ladders/)
---------------------
即玩具棋、蛇形棋，有的位置可以传送到下一个点。

```python
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)

        def rc(idx):
            r, c = (idx-1)//n, (idx-1)%n
            if r%2==1:
                c = n-1-c
            return n-1-r, c

        vis = set()
        q = deque([(1,0)])
        while q:
            idx, step = q.popleft()
            for i in range(1,7):
                next_idx = idx + i
                if next_idx > n*n: break

                nr, nc = rc(next_idx)
                if board[nr][nc] != -1:
                    next_idx = board[nr][nc]

                if next_idx == n * n:   # 到达终点
                    return step + 1

                if next_idx not in vis:
                    vis.add(next_idx)
                    q.append((next_idx, step + 1))   # 扩展新状态
        return -1

# 获取坐标、BFS、条件判断
```

[LCP 07. 传递信息](https://leetcode-cn.com/problems/chuan-di-xin-xi/)
--------------------

##### 1. DFS:

```python
class Solution:
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        d = defaultdict(list)
        for x, y in relation:
            d[x].append(y)

        def dfs(s, t, k):
            if k==1: 
                return (t in d[s])
            return sum([dfs(i, t, k-1) for i in d[s]])

        return int(dfs(0, n-1, k))
```

##### 2. DP:

```python
class Solution:
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        dp = [[0]*n for _ in range(k+1)]
        dp[0][0] = 1
        for i in range(1, k+1, 1):
            for x,y in relation:
                dp[i][y] += dp[i-1][x]
        return dp[k][n-1]
```

## [1034. 边界着色](https://leetcode-cn.com/problems/coloring-a-border/)

对联通分量的边界进行着色，并返回最终的 grid

```python
class Solution:
    def colorBorder(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0])
        border, vis = set(), set()
        mark = grid[row][col]

        def dfs(x, y):
            if not (0<=x<rows and 0<=y<cols and grid[x][y]==mark):
                return False
            if (x, y) in vis:
                return True
            vis.add((x,y))
            if dfs(x+1, y) + dfs(x-1, y) + dfs(x, y+1) + dfs(x, y-1) < 4:
                border.add((x,y))
            return True
        
        dfs(row, col)
        for x,y in border:
            grid[x][y] = color
        return grid

# 使用 border 存下来，并在最终时修改。
```


## [2045. 到达目的地的第二短时间](https://leetcode-cn.com/problems/second-minimum-time-to-reach-destination/)

最短路径问题 + 交通信号灯 + 第二小值

```python
# 总是时间少的先入队，不管有没有红灯
# 两个记录的BFS
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        g = defaultdict(list)
        for u,v in edges:
            g[u].append(v)
            g[v].append(u)
        
        # dp[i][0] 表示 1 到 i 的 最短路径
        # dp[i][1] 表示 1 到 i 的 次最短路径
        dist = [[float('inf')]*2 for _ in range(n+1)]
        dist[1][0] = 0
        q = deque([(1, 0)])   # n,t

        while dist[n][1] == float('inf'):
            u, t = q.popleft()
            for v in g[u]:
                d = t + 1
                if d < dist[v][0]:
                    dist[v][0] = d
                    q.append((v, d))
                elif dist[v][0] < d < dist[v][1]:
                    dist[v][1] = d
                    q.append((v, d))

        ans = 0
        for _ in range(dist[n][1]):
            if ans % (change*2) >= change:
                ans += change*2 - ans%(change*2)
            ans += time
        return ans
```