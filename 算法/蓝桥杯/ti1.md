题目来源：https://www.lanqiao.cn/contests/lqcup20/challenges/

## 排列小球

摆放三种颜色小球使其连续个数 严格递增

```python
l = [int(i) for i in input().split()]
n = sum(l)
res = 0

def dfs(n, cur, precolor):
    global res
    if n == 0:        # 唯一出口， 这是一个符合的排列
        res += 1
        return
    for i in range(3):
        if i == precolor:
            continue
        for num in range(cur+1, l[i]+1):
            l[i] -= num
            dfs(n-num, num, i)
            l[i] += num
            
dfs(n, 0, -1)
print(res)

```

## 扫地机器人

初看挺难的，从机器人的扫地范围入手，枚举或二分

```python
n, k = map(int, input().split())

l = []
for i in range(k):
    l.append(int(input()))
l.sort()

def check(l, x):
    cur = 0            # 当前扫地最大右边界
    for i in l:
        if i - x <= cur:
            cur = min(i, cur + 1) + x -1  # 最大范围变化
        else:
            return False
    if cur >= n:
        return True

res = 1
while(1):
    if check(l, res):
        break
    else:
        res += 1

print(2*(res-1))
```

## LCIS

二维 DP 打表，牛哇

```python
n, m = map(int, input().split())
l1 = [0] + list(map(int, input().split()))
l2 = [0] + list(map(int, input().split()))
dp = [[0]*(m+1) for _ in range(n+1)]

for i in range(1, n+1):
    mx = 1
    for j in range(1, m+1):
        dp[i][j] = dp[i-1][j]
        if l1[i] == l2[j]:
            dp[i][j] = max(mx, dp[i][j])
        elif l1[i] > l2[j]:
            mx = max(mx, dp[i-1][j] + 1)

print(max(dp[n]))
```

## 蓝桥幼儿园

基础并查集

```python
f = {}
def find(x):
    f.setdefault(x, x)
    while x != f[x]:
        f[x] = f[f[x]]
        x = f[x]
    return x
def union(x, y):
    f[find(x)] = find(y)
    
n, m = map(int, input().split())
for i in range(m):
    op, x, y = map(int, input().split())
    if op == 1:
        union(x, y)
    else:
        if find(x) == find(y):
            print("YES")
        else:
            print("NO")
```

## 字符串转换

编辑距离， leetcode 一道困难题

```python
s = list(input())
t = list(input())

def main(s, t):
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    for i in range(1, m+1):
        dp[i][0] = i
    for j in range(1, n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[-1][-1]

print(main(s, t))
```

## 编程作业

```python
# 不会
```