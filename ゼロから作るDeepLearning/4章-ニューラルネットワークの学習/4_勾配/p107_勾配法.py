import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # xと同じ形状のzerosを生成
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # calc f(x1,x2,...,idx+h,...,xn)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # calc f(x1,x2,...,idx-h,...,xn)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 元に戻す

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

if __name__ == '__main__':
    # f=sum(x*x) の最小値を勾配法で求めよ.
    def func(x):
        return sum(x * x)

    print(gradient_descent(func, np.array([-3.0, 4.0]), lr=0.1, step_num=100))

    # 学習率が小さいとうまくいかない.

    print(gradient_descent(func, np.array([-3.0, 4.0]), lr=10.0, step_num=100))
    print(gradient_descent(func, np.array([-3.0, 4.0]), lr=1e-10, step_num=100))

