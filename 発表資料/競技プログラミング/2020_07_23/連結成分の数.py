import sys
sys.setrecursionlimit(1000000000) # Pythonは標準で2000回までしか再帰できない.
# 深さ優先探索
def dfs(v):
    Not_using[v] = False
    for elem in es[v]:
        if Not_using[elem]:
            dfs(elem)

# 入力
N, M = map(int, input().split())
es = [[] for _ in range(N)] # 隣接リスト
for _ in range(M):
    a, b = map(int, input().split())
    es[a - 1].append(b - 1)
    es[b - 1].append(a - 1)
Not_using = [True] * N

# 連結成分の数
cnt = 0
for elem in range(N):
    if Not_using[elem]:
        cnt += 1
        dfs(elem)

print(cnt)

# O(N+M)