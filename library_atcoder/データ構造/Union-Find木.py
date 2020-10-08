class UnionFind:
    __slots__ = ["N", "root"]

    def __init__(self, N):
        """
        N:要素数
        root:各要素の親要素の番号を格納するリスト.
             ただし, root[x] < 0 ならその頂点が根で-root[x]が木の要素数.
        rank:ランク
        """
        self.N = N
        self.root = [-1] * N

    def __repr__(self):
        return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

    def find(self, x):
        """頂点xの根を見つける"""
        while self.root[x] >= 0:
            x = self.root[x]
        return x

    def union(self, x, y):
        """x, yが属する木をunion"""
        # 根を比較する
        # すでに同じ木に属していた場合は何もしない.
        # 違う木に属していた場合はrankを見てくっつける方を決める.
        # rankが同じ時はrankを1増やす
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        elif self.root[y] < self.root[x]:
            x, y = y, x
        self.root[x] += self.root[y]
        self.root[y] = x

    def same(self, x, y):
        """xとyが同じグループに属するかどうか"""
        return self.find(x) == self.find(y)

    def count(self, x):
        """頂点xが属する木のサイズを返す"""
        return - self.root[self.find(x)]

    def members(self, x):
        """xが属する木の要素を列挙"""
        _root = self.find(x)
        return [i for i in range(self.N) if self.find(i) == _root]

    def roots(self):
        """森の根を列挙"""
        return [i for i, x in enumerate(self.root) if x < 0]

    def group_count(self):
        """連結成分の数"""
        return len(self.roots())

    def all_group_members(self):
        """{ルート要素: [そのグループに含まれる要素のリスト], ...}の辞書を返す"""
        return {r: self.members(r) for r in self.roots()}
