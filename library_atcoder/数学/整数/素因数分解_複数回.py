from math import sqrt


class factorization:
    def __init__(self, n=1_000_000_000_000):
        """default:n=10**12"""
        n = sqrt(n) + 1
        n, correction = n - n % 6 + 6, 2 - (n % 6 > 1)
        sieve = [True] * (n // 3)
        for i in range(1, int(sqrt(n)) // 3 + 1):
            if sieve[i]:
                k = 3 * i + 1 | 1
                sieve[k * k // 3::2 * k] = [False] * ((n // 6 - k * k // 6 - 1) // k + 1)
                sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = [False] * ((n // 6 - k * (k - 2 * (i & 1) + 4) // 6 - 1) // k + 1)

        self.sieve = [2, 3] + [3 * i + 1 | 1 for i in range(1, n // 3 - correction) if sieve[i]]

    def fact(self, n):
        if n == 1:
            return [[1]]
        arr = []
        temp = n
        for i in self.sieve:
            if temp % i == 0:
                cnt = 0
                while temp % i == 0:
                    cnt += 1
                    temp //= i
                arr.append([i, cnt])
        if temp != 1:
            arr.append([temp, 1])
        return arr
