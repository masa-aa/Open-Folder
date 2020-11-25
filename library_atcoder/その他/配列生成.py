def box(*dims, initial=0):
    """初期値initialでdim[0]*dim[1]*...*dim[n]の配列を生成"""
    n = len(dims)
    # [[[... + [init] * m_0 + for _ in range(m_i)]...]...]...] の文字を生成
    s = "[" * n + "{}] * {}" + " for _ in range({})]" * (n - 1)
    # evalする
    # print(s.format(initial, *dims))
    return eval(s.format(initial, *dims))


if __name__ == "__main__":

    a = box(2, 2, 3, initial=1)
    a[2][1][0] = 10
    print(a)

    b = box(3, 3)
    print(b)
