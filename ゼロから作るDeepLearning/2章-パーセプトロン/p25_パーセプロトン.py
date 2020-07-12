"""
パーセプロトン:y=(w1*x1+w2*x2>theta)
"""
def per(x1,x2,w1,w2,theta):
    return w1*x1+w2*x2 > theta

# AND回路
def AND(x1,x2):
    return per(x1,x2,0.5,0.5,0.7)
print(AND(0,0),AND(0,1),AND(1,0),AND(1,1))