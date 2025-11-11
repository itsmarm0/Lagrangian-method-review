import numpy as np
from scipy.optimize import fsolve

def func(X) :
    x = X[0]
    y = X[1]
    z = X[2]
    l1 = X[3] 
    l2 = X[4] 

    return x + y + z + l1 * (x**2 + y - 3) + l2 * (x + 3*y +2*z - 7)

def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3 
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda

X1 = fsolve(dfunc, [0, 1, 0, 0, 0])
print (X1, func(X1))

X2 = fsolve(dfunc, [0, -1, 0, 0, 0])
print (X2, func(X2))