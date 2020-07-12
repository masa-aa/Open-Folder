# 普通の実装
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0

# numpy対応
import numpy as np
def step_function(x):
    y = x > 0
    return y.astype(np.int) # bool -> int