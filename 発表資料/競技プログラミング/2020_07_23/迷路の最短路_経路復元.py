from collections import deque 

if __name__ == '__main__':

    N, M = map(int, input().split())
    s, t = map(int, input().split())
    s, t = s - 1, t - 1
    es = [[] for _ in range(N)] # 隣接リスト
    for _ in range(M):
        a, b = map(int, input().split())
        es[a - 1].append(b - 1)
        es[b - 1].append(a - 1)

    # 距離を更新するリスト
    q = deque([s])
    # sからの距離
    INF = 10 ** 9
    d = [INF] * N
    d[s] = 0
    prev = [-1] * N

    while q:
        v = q.popleft()
        if v == t:
            break
        for elem in es[v]:
            if d[elem] == INF:
                q.append(elem)
                d[elem] = d[v] + 1
                prev[elem] = v
                
    
    print(d[t])
    path = [t + 1]
    v = t
    while v != s:
        v = prev[v]
        path.append(v + 1)
    path.reverse()
    print(*path,sep=" -> ")
# O(N+M)