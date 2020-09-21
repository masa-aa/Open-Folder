from collections import Counter
from functools import reduce
from itertools import product
from math import gcd
from operator import mul
from random import randrange


class PrimeFactor:
    def __init__(self):
        pass

    def is_prime(self, N):
        """ミラーラビン"""
        v = [2, 7, 61] if N < 4_759_123_141 else \
            [2, 3, 5, 7, 11, 13, 17] if N < 341_550_071_728_321 else \
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        if N < 2:
            return False
        if N in v:
            return True
        d = N - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1

        for a in v:
            if pow(a, d, N) != 1:
                ok = True
                for r in range(s):
                    if pow(a, d * 1 << r, N) == N - 1:
                        ok = 0
                        break
                if ok:
                    return False
        return True

    def f(self, x, n, c):
        return (x * x + c) % n

    def get_factor(self, n):
        c = randrange(1, n)
        if n % 2 == 0:
            return 2
        if self.is_prime(n):
            return n
        x = randrange(1, n)
        y, d = x, 1

        while d == 1:
            x = self.f(x, n, c)
            y = self.f(self.f(y, n, c), n, c)
            d = gcd(abs(x - y), n)
        if d == n:
            return self.get_factor(n)
        return self.get_factor(d)

    def factorization(self, N):
        """素因数分解"""
        if N == 1:
            return [1]
        arr = []
        while N > 1:
            k = self.get_factor(N)
            cnt = 0
            while N % k == 0:
                cnt += 1
                N //= k
                arr.append(k)
        return arr

    def divisors(self, N):
        """約数列挙"""
        if N == 1:
            return [1]
        c = Counter(self.factorization(N))
        mul_list = []
        for i in c:
            tmp = [1] * (c[i] + 1)
            for j in range(c[i]):
                tmp[j + 1] = tmp[j] * i
            mul_list.append(tmp)
        res = [reduce(mul, v) for v in product(*mul_list)]
        return res

    def count_divisors(self, N):
        """約数の個数"""
        if N == 1:
            return 1
        res = 1
        for i in Counter(self.factorization(N)).values():
            res *= i + 1
        return res
