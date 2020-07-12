import numpy as np
x = np.array([1.0,2.0,3.0])
print(x)
print(type(x))

# 計算
x=np.array([1.0,2.0,3.0])
y=2*x # [2. 4. 6.]
print(x+y)
print(x-y)
print(x*y)
print(x/y)

# N次元配列
A = np.array([[1,2],[3,4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3,0],[0,6]])
print(A+B)
print(A*B)
print(10*A)

# broadcast
A = np.array([[1,2],[3,4]])
B = np.array([10,20])
print(A*B)

# access
X = np.array([[51,55],[14,19],[0,4]])
print(X)
print(X[0])
print(X[0][1])
print(X[0,1])

for v in X:
    print(v)

X=X.flatten() # 1次元化
print(X[np.array([0,2,4])]) #0,2,4番目のarrayを返す.

print(X>15)
print(X[X>15]) # 15より大きい要素
