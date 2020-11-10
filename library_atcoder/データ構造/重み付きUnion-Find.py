class WeightedUnionFind:
    def __init__(self, n):
        """root[v] = vの親, rank[v] = 木の高さ, diff_weight[v] = 根からの重み"""
        self.root = list(range(n))
        self.rank = [0] * n
        self.diff_weight = [0] * n

    def find(self, x):
        stack = []
        while self.root[x] != x:
            stack.append(x)
            x = self.root[x]

        stack.reverse()
        for i in stack:
            self.diff_weight[i] += self.diff_weight[self.root[i]]
            self.root[i] = x

        return x

    def weight(self, x):
        self.find(x)
        return self.diff_weight[x]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def union(self, x, y, w):
        """weight(y) - weight(x) = w となるように union する"""
        w += self.weight(x)
        w -= self.weight(y)
        x, y = self.find(x), self.find(y)
        if x == y:
            return
        if self.rank[x] < self.rank[y]:
            x, y, w = y, x, -w
        if x == y:
            self.rank[x] += 1
        self.root[y] = x
        self.diff_weight[y] = w


# verify
# https://onlinejudge.u-aizu.ac.jp/solutions/problem/DSL_1_B/review/4984129/masa_aa/Python3
