import numpy as np
from scipy.optimize import fsolve

def func(X) :
    x = X[0]
    y = X[1]
    z = X[2]
    l1 = X[3] # This is the first multiplier
    l2 = X[4] # This is the second multiplier

    return x + y + z + l1 * (x**2 + y - 3) + l2 * (x + 3*y +2*z - 7)

def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3 # This is the step size used in the finite difference
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda

# This is the max
X1 = fsolve(dfunc, [0, 1, 0, 0, 0])
print (X1, func(X1))

# This is the min
X2 = fsolve(dfunc, [0, -1, 0, 0, 0])
print (X2, func(X2))