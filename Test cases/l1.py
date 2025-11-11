import numpy as np
from scipy.optimize import fsolve

def func(X) :
    x = X[0]
    y = X[1]
    l = X[2] 

    return x + y + l * (x**2 + y**2 - 1)

def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3 
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda

X1 = fsolve(dfunc, [1, 1, 0])
print (X1, func(X1))

X2 = fsolve(dfunc, [-1, -1, 0])
print (X2, func(X2))