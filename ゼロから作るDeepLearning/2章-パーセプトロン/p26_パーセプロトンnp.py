"""
パーセプロトン:y=(b+w1*x1+w2*x2>0)
"""
import numpy as np
def per(x1,x2,w1,w2,b): # w:重み b:バイアス
    x=np.array([x1,x2])
    w=np.array([w1,w2])
    return b+sum(x*w)>0

# AND回路
def AND(x1,x2):
    return per(x1,x2,0.5,0.5,-0.7)
print(AND(0,0),AND(0,1),AND(1,0),AND(1,1))

# NAND回路
def NAND(x1,x2):
    return per(x1,x2,-0.5,-0.5,0.7)
print(NAND(0,0),NAND(0,1),NAND(1,0),NAND(1,1))

# or回路
def OR(x1,x2):
    return per(x1,x2,0.5,0.5,-0.3)
print(OR(0,0),OR(0,1),OR(1,0),OR(1,1))

def XOR(x1,x2):
    return AND(NAND(x1,x2),OR(x1,x2))
print(XOR(0,0),XOR(0,1),XOR(1,0),XOR(1,1))