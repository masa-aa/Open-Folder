# nが大きくてkが小さい O(k)


def comb(n, k, mod=1000000007):
    ans = 1
    inv = 1
    for i in range(1, k + 1):
        ans = ans * (n - k + i) % mod
        inv = inv * pow(i, mod - 2, mod) % mod
    return ans * inv % mod
