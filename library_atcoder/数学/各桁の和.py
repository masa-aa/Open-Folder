"""
各桁の和を求める
"""


def dgsm(n):
    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return(ans)

# print(dgsm(1342))
# 1+3+4+2=10
