class matrix:
    def __init__(self, A: "N * M Matrix"):
        self.__A = A
        self.row = len(A)
        self.column = len(A[0])

    def __getitem__(self, item):
        return self.__A[item[0]][item[1]]

    def __setitem__(self, key, value):
        self.__A[key[0]][key[1]] = value

    def __repr__(self):
        return "\n".join(" ".join(map(str, v)) for v in self.__A)

    # eq(==), ne(!=)
    def __eq__(self, other):
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
        res = [[0] * other.column for i in range(self.row)]
        for row in range(self.row):
            for column in range(self.column):
                res[row][column] = self.__A[row][column] + other[row, column]
        return self.__class__(res)

    def __iadd__(self, other):
        for row in range(self.row):
            for column in range(self.column):
                self.__A[row][column] += other[row, column]
        return self

    # sub(-)
    def __sub__(self, other):
        res = [[0] * other.column for i in range(self.row)]
        for row in range(self.row):
            for column in range(self.column):
                res[row][column] = self.__A[row][column] - other[row, column]
        return self.__class__(res)

    def __isub__(self, other):
        for row in range(self.row):
            for column in range(self.column):
                self.__A[row][column] -= other[row, column]
        return self

    # mul(*)
    def __mul__(self, other):
        if isinstance(other, int):
            res = [[0] * self.column for i in range(self.row)]
            for row in range(self.row):
                for column in range(self.column):
                    res[row][column] = other * self.__A[row][column]

        else:
            res = [[0] * other.column for i in range(self.row)]
            for i in range(self.row):
                for j in range(other.column):
                    for k in range(self.column):
                        res[i][j] += self.__A[i][k] * other[k, j]
        return self.__class__(res)

    def __imul__(self, other):
        if isinstance(other, int):
            for row in range(self.row):
                for column in range(self.column):
                    self.__A[row][column] *= other
            return self

        res = [[0] * other.column for i in range(self.row)]
        for i in range(self.row):
            for j in range(other.column):
                for k in range(self.column):
                    res[i][j] += self.__A[i][k] * other[k, j]
        return self.__class__(res)

    def __rmul__(self, other):
        res = [[0] * self.column for i in range(self.row)]
        for row in range(self.row):
            for column in range(self.column):
                res[row][column] = other * self.__A[row][column]
        return self.__class__(res)

    # pow(**)
    def __pow__(self, other):
        res = identity(self.row)
        tmp = matrix(self.__A)
        for i in range(other.bit_length()):
            if (other >> i) & 1:
                res *= tmp
            tmp *= tmp
        return res

    # neg(-A), pos(+A)
    def __neg__(self):
        res = matrix(self.__A)
        for i in range(self.row):
            for j in range(self.column):
                res[i, j] = -res[i, j]
        return res

    def __pos__(self):
        return self


def identity(N):
    """N*Nの単位行列を返す"""
    res = [[0] * N for _ in range(N)]
    for i in range(N):
        res[i][i] = 1
    return matrix(res)


def zeros(N):
    """N*Nの0行列を返す"""
    return matrix([[0] * N for _ in range(N)])


if __name__ == "__main__":
    A = matrix([[2, 0], [0, 2]])
    B = matrix([[1, 2], [2, 2]])
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
