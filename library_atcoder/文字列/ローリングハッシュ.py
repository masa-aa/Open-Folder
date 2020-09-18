from random import randint


class RollingHash:
    def __init__(self, s):
        self.base = [7073, 7577, 5445, 2742, 6972, 7547, 2267, 286, 6396, 7147,
                     3307, 188, 266, 8253, 2818, 9527, 5110, 1207, 4633, 6196,
                     309, 2646, 7533, 85, 9870, 4730, 6862, 9213, 7456, 7098,
                     6805, 674, 5821, 4864, 8061, 1826, 2219, 459, 5937, 5667,
                     9033, 5552, 7263, 2402, 9809, 3701, 7048, 2874, 8350, 6006,
                     973, 3317, 2522, 5546, 1669, 1545, 7972, 4979, 9905, 173,
                     6812, 7715, 5006, 6068, 6340, 4989, 5510, 6380, 1200, 6739,
                     5527, 4000, 6519, 3448, 2933, 6048, 3133, 1667, 9086, 8368,
                     4914, 7142, 2770, 7752, 391, 7052, 5476, 3105, 8322, 3501,
                     7454, 3167, 8730, 9002, 4564, 138, 2197, 7238, 3411, 7433][randint(0, 100)]
        self.mod = 4611686018427387903
        self.size = len(s)
        self.string = s

        self.hash = self.make_hashtable(s)
        self.pow = self.make_powtable()

    def make_hashtable(self, _s):
        hashtable = [0] * (self.size + 1)
        for i in range(self.size):
            hashtable[i + 1] = (hashtable[i] * self.base + ord(_s[i])) % self.mod
        return hashtable

    def make_powtable(self):
        power = [1] * (self.size + 1)
        for i in range(self.size):
            power[i + 1] = (self.base * power[i]) % self.mod
        return power

    def get_hash(self, left, right):
        """get hash of s[left:right]"""
        return (self.hash[right] - self.hash[left] * self.pow[right - left]) % self.mod

    def contain(self, a):
        """return a in s"""
        m = len(a)
        if m > self.size:
            return False
        hashs = self.get_hash(0, m)
        hasha = 0
        for i in range(m):
            hasha = (hasha * self.base + ord(a[i])) % self.mod
        for i in range(self.size - m + 1):
            if hasha == hashs:
                return True
            hashs = self.get_hash(i, m + i)
        return hasha == hashs
