import numpy as np
from scipy.optimize import fsolve

def func(X) :
    x = X[0]
    y = X[1]
    z = X[2]
    l = X[3] 
    return 2*x + x*y + 3*y + l * (x**2 + y - z**2 - 3)

def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda


X1_1 = fsolve(dfunc, [2, 2, -1, 0])
print (X1_1, func(X1_1))

X1_2 = fsolve(dfunc, [2, 2, 0, 1])
print (X1_2, func(X1_2))

X2_1 = fsolve(dfunc, [-2, -2, -1, 0])
print (X2_1, func(X2_1))

X2_2 = fsolve(dfunc, [-2, -2, 0, 1])
print (X2_2, func(X2_2))