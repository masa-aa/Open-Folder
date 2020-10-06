class mod_matrix:
    def __init__(self, A: "N * M Matrix", mod=1000000007, is_mod=True):
        """is_mod 既にmodが取られているか否か"""
        self.__A = A
        self.row = len(A)
        self.column = len(A[0])
        self.mod = mod
        if not is_mod:
            for row in range(self.row):
                for column in range(self.column):
                    self.__A[row][column] %= self.mod

    def __getitem__(self, item):
        return self.__A[item[0]][item[1]]

    def __setitem__(self, key, value):
        self.__A[key[0]][key[1]] = value

    def __repr__(self):
        return "\n".join(" ".join(map(str, v)) for v in self.__A)

    # eq(==), ne(!=)
    def __eq__(self, other):
        """任意の要素が同じで同じmodのとき=="""
        if self.mod != other.mod:
            return False
        for row in range(self.row):
            for column in range(self.column):
                if self.__A[row][column] != other[row, column]:
                    return False
        return True

    def __ne__(self, other):
        for row in range(self.row):
            for column in range(self.column):
                if self.__A[row][column] != other[row, column]:
                    return True
        return False

    # add(+)
    def __add__(self, other):
        """A + B, AとBの型が違うとき未定義"""
        res = [[0] * other.column for i in range(self.row)]
        for row in range(self.row):
            for column in range(self.column):
                res[row][column] = self.__A[row][column] + other[row, column]
                if res[row][column] >= self.mod:
                    res[row][column] -= self.mod
        return self.__class__(res)

    def __iadd__(self, other):
        for row in range(self.row):
            for column in range(self.column):
                self.__A[row][column] += other[row, column]
                if self.__A[row][column] >= self.mod:
                    self.__A[row][column] -= self.mod
        return self

    # sub(-)
    def __sub__(self, other):
        """A - B, AとBの型が違うとき未定義"""
        res = [[0] * other.column for i in range(self.row)]
        for row in range(self.row):
            for column in range(self.column):
                res[row][column] = self.__A[row][column] - other[row, column]
                if res[row][column] < 0:
                    res[row][column] += self.mod
        return self.__class__(res)

    def __isub__(self, other):
        for row in range(self.row):
            for column in range(self.column):
                self.__A[row][column] -= other[row, column]
                if self.__A[row][column] < 0:
                    self.__A[row][column] += self.mod
        return self

    # mul(*)
    def __mul__(self, other):
        """N*M行列AとM*L行列Bに対してN*L行列A*Bまたは
        scala値aとM*L行列Bに対してM*L行列a*Bを定義
        それ以外は未定義"""
        if isinstance(other, mod_matrix):
            res = [[0] * other.column for i in range(self.row)]
            for i in range(self.row):
                for j in range(other.column):
                    for k in range(self.column):
                        res[i][j] += self.__A[i][k] * other[k, j]
                    res[i][j] %= self.mod

        else:
            res = [[0] * self.column for i in range(self.row)]
            for row in range(self.row):
                for column in range(self.column):
                    res[row][column] = other * self.__A[row][column]
                    res[row][column] %= self.mod
        return self.__class__(res)

    def __imul__(self, other):
        if isinstance(other, mod_matrix):
            res = [[0] * other.column for i in range(self.row)]
            for i in range(self.row):
                for j in range(other.column):
                    for k in range(self.column):
                        res[i][j] += self.__A[i][k] * other[k, j]
                    res[i][j] %= self.mod
            return self.__class__(res)

        for row in range(self.row):
            for column in range(self.column):
                self.__A[row][column] *= other
                self.__A[row][column] %= self.mod
        return self

    def __rmul__(self, other):
        """scala値aとM*L行列Bに対してM*L行列B*aを定義, それ以外は未定義"""
        res = [[0] * self.column for i in range(self.row)]
        for row in range(self.row):
            for column in range(self.column):
                res[row][column] = other * self.__A[row][column]
                res[row][column] %= self.mod
        return self.__class__(res)

    # pow(**)
    def __pow__(self, other):
        """N*N行列Aに対してA**nを定義"""
        res = identity(self.row, self.mod)
        tmp = mod_matrix(self.__A, self.mod)
        for i in range(other.bit_length()):
            if (other >> i) & 1:
                res *= tmp
            tmp *= tmp
        return res

    # neg(-A), pos(+A)
    def __neg__(self):
        res = mod_matrix(self.__A, self.mod)
        for i in range(self.row):
            for j in range(self.column):
                res[i, j] = -res[i, j]
                res[i, j] %= self.mod
        return res

    def __pos__(self):
        return self


def identity(N, mod):
    """N*Nの単位行列を返す"""
    res = [[0] * N for _ in range(N)]
    for i in range(N):
        res[i][i] = 1
    return mod_matrix(res, mod)


def zeros(N, mod):
    """N*Nの0行列を返す"""
    return mod_matrix([[0] * N for _ in range(N)], mod)


if __name__ == "__main__":
    mod = 100
    A = mod_matrix([[50, 0], [0, 50]], mod)
    B = mod_matrix([[1, 3], [3, 2]], mod)
    print("A")
    print(A)
    print("\nB")
    print(B)
    print("\nA + B")
    print(A + B)
    print("\nA - B")
    print(A - B)
    print("\nA * B")
    print(A * B)
    # print("\nA += B")
    # A += B
    # print(A)
    # print("\nA *= B")
    # A *= B
    # print(A)
    # print(A == matrix([[1, 0], [0, 1]]))
    print("\nA ** 2")
    print(A ** 5)
    print("\n3 * A")
    print(3 * A)

    A = mod_matrix([[50, 0], [0, 50]], mod)
    B = mod_matrix([[50, 0], [0, 50]], mod + 1)
    print(A == B)
